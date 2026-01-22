"""Training script with Hydra config and W&B logging."""

import os
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from arxiv_classifier.data import ArxivDataset, load_split
from arxiv_classifier.models import get_model
from arxiv_classifier.training_output import (
    ProgressBar,
    log_epoch_summary,
    log_training_complete,
    log_training_config,
    set_seed,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text_collate_fn(batch: list[dict]) -> dict:
    """Simple collator that passes texts through."""
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    return {"texts": texts, "labels": labels}


def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, epoch, epochs, cfg):
    """Train for one epoch."""
    model.train()

    pbar = ProgressBar(
        total=len(dataloader.dataset),
        desc=f"E{epoch + 1}/{epochs}",
        device=DEVICE,
        batch_size=cfg.training.batch_size,
    )

    for batch in dataloader:
        texts = batch["texts"]
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            logits = model(texts)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if pbar.batch_count > 1:  # Skip first batch
            scheduler.step()

        pbar.update(loss, labels.size(0))

    avg_loss = pbar.get_loss()
    pbar.close()
    return avg_loss


def validate(model, dataloader, criterion, cfg):
    """Validate and return loss and accuracy."""
    model.eval()

    pbar = ProgressBar(
        total=len(dataloader.dataset),
        desc="Valid",
        device=DEVICE,
        batch_size=cfg.training.batch_size,
        is_eval=True,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            texts = batch["texts"]
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
                logits = model(texts)
                loss = criterion(logits, labels)

            pbar.update(loss, labels.size(0))

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = pbar.get_loss()
    pbar.close()
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(cfg: DictConfig, base_dir: Path | None = None):
    """Main training function.

    Args:
        cfg: Hydra configuration
        base_dir: Base directory for data/output paths. If None, uses Hydra's original cwd.
    """
    set_seed(cfg.training.seed)

    # Resolve base directory (Hydra changes cwd, Modal uses explicit path)
    if base_dir is None:
        base_dir = Path(hydra.utils.get_original_cwd())

    output_dir = base_dir / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    data_dir = base_dir / "data" / "processed"
    dataset = load_split("train", data_dir=data_dir)
    val_dataset = load_split("val", data_dir=data_dir)

    # Subsample if requested
    if cfg.training.max_samples:
        max_samples = cfg.training.max_samples
        if len(dataset) > max_samples:
            indices = torch.randperm(len(dataset))[:max_samples].tolist()
            dataset = ArxivDataset(
                [dataset.texts[i] for i in indices],
                dataset.labels[indices],
            )
        # Subsample val proportionally (use same size as train for tiny datasets)
        val_samples = max_samples if max_samples < 1000 else max(1000, max_samples // 7)
        if len(val_dataset) > val_samples:
            indices = torch.randperm(len(val_dataset))[:val_samples].tolist()
            val_dataset = ArxivDataset(
                [val_dataset.texts[i] for i in indices],
                val_dataset.labels[indices],
            )

    train_size = len(dataset)
    val_size = len(val_dataset)

    # Create dataloaders
    num_workers = min(cfg.training.num_workers, os.cpu_count() or 1)
    loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "collate_fn": text_collate_fn,
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    train_loader = DataLoader(dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    log_training_config(train_size, val_size, DEVICE, num_workers, cfg.training.batch_size)

    # Create model from config
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "name"}
    model = get_model(cfg.model.name, **model_kwargs).to(DEVICE)
    trainable, total = model.count_parameters()
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({trainable / total:.1%})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=0.01,
    )

    # Scheduler
    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(cfg.training.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, epoch, cfg.training.epochs, cfg
        )
        val_loss, val_acc = validate(model, val_loader, criterion, cfg)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        log_epoch_summary(epoch, cfg.training.epochs, train_loss, val_loss, val_acc, improved)

        # Log to W&B
        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    log_training_complete(output_dir)

    # Log best metrics to W&B summary
    if wandb.run is not None:
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_val_accuracy"] = best_val_acc

    return model


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Entry point with Hydra config."""
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Initialize W&B
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        name=f"{cfg.model.name}",
    )

    try:
        model = train(cfg)
    finally:
        wandb.finish()

    return model


if __name__ == "__main__":
    main()
