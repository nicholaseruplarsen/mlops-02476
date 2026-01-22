import os

from invoke.context import Context
from invoke.tasks import task

WINDOWS = os.name == "nt"
PROJECT_NAME = "arxiv_classifier"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context, max_samples: int | None = None) -> None:
    """Preprocess data."""
    cmd = f"uv run python -m {PROJECT_NAME}.data"
    if max_samples:
        cmd += f" --max-samples {max_samples}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def train(
    ctx: Context,
    experiment: str | None = None,
    max_samples: int | None = None,
    epochs: int | None = None,
    wandb_mode: str = "online",
    modal: bool = False,
    extra: str | None = None,
) -> None:
    if modal:
        cmd = f"uv run modal run -m {PROJECT_NAME}.modal_train"
        if experiment:
            cmd += f" --experiment {experiment}"
        if max_samples:
            cmd += f" --max-samples {max_samples}"
        if epochs:
            cmd += f" --epochs {epochs}"
        if wandb_mode != "online":
            cmd += f" --wandb-mode {wandb_mode}"
    else:
        cmd = "uv run"
        if extra:
            cmd += f" --extra {extra}"
        cmd += f" python -m {PROJECT_NAME}.train"
        if experiment:
            cmd += f" experiment={experiment}"
        if max_samples:
            cmd += f" training.max_samples={max_samples}"
        if epochs:
            cmd += f" training.epochs={epochs}"
        if wandb_mode != "online":
            cmd += f" wandb.mode={wandb_mode}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
