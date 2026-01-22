"""Modal cloud training for arXiv classifier.

Usage:
    # Run on Modal cloud
    modal run -m arxiv_classifier.modal_train

    # With experiment override
    modal run -m arxiv_classifier.modal_train --experiment sentence_transformer

    # With all options
    modal run -m arxiv_classifier.modal_train --experiment scibert_full --epochs 5 --max-samples 10000

Note: GPU type is configured in the @app.function decorator (default: A10G).
      To change GPU, modify the `gpu` parameter in the decorator.
"""

import modal

app = modal.App("arxiv-classifier-training")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "wandb",
        "hydra-core",
        "omegaconf",
        "sentence-transformers",
        "nvitop",
        "typer",
        "matplotlib",
        "loguru",
        "peft",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data/processed", remote_path="/root/data/processed")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_fn(
    experiment: str = "scibert_frozen",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_samples: int | None = None,
    wandb_mode: str = "online",
    seed: int = 42,
) -> dict:
    """Training function that runs on Modal."""
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Build Hydra overrides
    overrides = [
        f"experiment={experiment}",
        f"wandb.mode={wandb_mode}",
        f"training.seed={seed}",
        "output_dir=/root/models",
    ]
    if epochs is not None:
        overrides.append(f"training.epochs={epochs}")
    if batch_size is not None:
        overrides.append(f"training.batch_size={batch_size}")
    if learning_rate is not None:
        overrides.append(f"training.learning_rate={learning_rate}")
    if max_samples is not None:
        overrides.append(f"training.max_samples={max_samples}")

    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir="/root/configs", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

        import wandb
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg))

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            name=f"{cfg.model.name}-modal",
        )

        try:
            from pathlib import Path

            from arxiv_classifier.train import train as do_train

            do_train(cfg, base_dir=Path("/root"))
        finally:
            wandb.finish()

    return {"status": "completed", "experiment": experiment}


@app.local_entrypoint()
def main(
    experiment: str = "scibert_frozen",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_samples: int | None = None,
    wandb_mode: str = "online",
    seed: int = 42,
):
    """Local entrypoint that dispatches to Modal.

    Args:
        experiment: Experiment config (scibert_frozen, scibert_full, sentence_transformer)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_samples: Max training samples for quick iteration
        wandb_mode: W&B mode (online, offline, disabled)
        seed: Random seed
    """
    result = train_fn.remote(
        experiment=experiment,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_samples=max_samples,
        wandb_mode=wandb_mode,
        seed=seed,
    )
    print(f"Training completed: {result}")
