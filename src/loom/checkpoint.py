from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torchdata.nodes import Loader

    from loom.runner import RunConfig
    from loom.task import Task


def atomic_save(obj: object, path: Path) -> None:
    """Write `obj` to `path` via a temp file and `os.replace`: bytes raw, else `torch.save`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            if isinstance(obj, bytes):
                f.write(obj)
            else:
                torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    for stale in path.parent.glob(re.sub(r"\d+", "*", path.name) + ".tmp"):
        stale.unlink(missing_ok=True)


def _jsonable_config(config: RunConfig) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(config), default=str))


def save_checkpoint(
    directory: Path,
    *,
    step: int,
    epoch: int,
    in_epoch: bool,
    task: Task,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    train_loader: Loader,
    config: RunConfig,
    device: torch.device,
) -> Path:
    """Atomically write `directory/step-{step:08d}.pt`; `epoch` counts closed epochs."""
    path = Path(directory) / f"step-{step:08d}.pt"
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    ckpt = {
        "format_version": 1,
        "loom_version": importlib.metadata.version("loom"),
        "torch_version": str(torch.__version__),
        "step": step,
        "epoch": epoch,
        "in_epoch": in_epoch,
        "task": task.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "train_loader": train_loader.state_dict(),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": cuda,
        },
        "config": _jsonable_config(config),
    }
    atomic_save(ckpt, path)
    return path


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint dict on CPU with `weights_only=True`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format_version") != 1:
        raise ValueError(f"{path} is not a loom checkpoint with format_version 1")
    return ckpt


def _checkpoint_steps(directory: str | Path) -> dict[int, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {
        int(m[1]): p
        for p in directory.iterdir()
        if (m := re.fullmatch(r"step-(\d+)\.pt", p.name))
    }


def latest_checkpoint(directory: str | Path) -> Path | None:
    """The `step-*.pt` in `directory` with the highest step, or None."""
    steps = _checkpoint_steps(directory)
    return steps[max(steps)] if steps else None


def restore_rng(rng: dict, device: torch.device) -> None:
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"], device)
