import logging
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from loom.data import DataFeed, _to_loader
from loom.task import Task

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    """Every Runner knob; validated on construction."""

    max_epochs: int | None = None
    max_steps: int | None = None
    grad_accum_steps: int = 1
    max_grad_norm: float | None = None  # None = no clipping
    precision: Literal["fp32", "bf16"] = "fp32"
    device: str | None = None  # None = cuda if available else cpu
    seed: int = 0
    log_every_n_steps: int = 50
    eval_every_n_epochs: int | None = 1
    eval_every_n_steps: int | None = None
    checkpoint_dir: str | Path | None = None  # None = no checkpoints
    checkpoint_every_n_steps: int | None = None
    resume_from: str | Path | None = None  # path, or "latest"

    def __post_init__(self) -> None:
        for name in (
            "max_epochs",
            "max_steps",
            "grad_accum_steps",
            "log_every_n_steps",
            "eval_every_n_epochs",
            "eval_every_n_steps",
            "checkpoint_every_n_steps",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 when set, got {value}")
        if self.max_grad_norm is not None and not self.max_grad_norm > 0:
            raise ValueError(
                f"max_grad_norm must be > 0 when set, got {self.max_grad_norm}"
            )
        if self.max_epochs is None and self.max_steps is None:
            raise ValueError("set max_epochs or max_steps; both are None")
        if self.precision == "fp16":
            raise ValueError("fp16 needs GradScaler; use bf16")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError(
                f"precision must be 'fp32' or 'bf16', got {self.precision!r}"
            )
        if self.checkpoint_dir is None:
            if self.resume_from == "latest":
                raise ValueError("resume_from='latest' needs checkpoint_dir")
            if self.checkpoint_every_n_steps is not None:
                raise ValueError("checkpoint_every_n_steps needs checkpoint_dir")


def seed(seed: int) -> None:
    """Seed Python's `random` and torch (all devices)."""
    random.seed(seed)
    torch.manual_seed(seed)


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=device.type == "cuda")
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(_to_device(v, device) for v in obj))
    if isinstance(obj, list | tuple):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _require(task: Task, methods: tuple[str, ...], entry: str) -> None:
    for m in methods:
        if getattr(type(task), m) is getattr(Task, m):
            raise TypeError(
                f"{type(task).__name__} must override {m} to use Runner.{entry}"
            )


class Runner:
    """Stateless loop driver: owns device, precision, accumulation and cadences."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        if config.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(f"device={config.device!r} but CUDA is not available")
            if device.index is not None:
                torch.cuda.set_device(device)
        self.device = device
        loom_logger = logging.getLogger("loom")
        if not loom_logger.handlers and not logging.getLogger().handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            handler.setLevel(logging.INFO)
            loom_logger.addHandler(handler)
            loom_logger.setLevel(logging.INFO)

    def fit(self, task: Task, data: DataFeed) -> dict[str, float]:
        cfg = self.config
        start = time.perf_counter()
        seed(cfg.seed)
        _require(task, ("train_step", "build_optimizer"), "fit")
        task = self._setup(task)
        opt = task.build_optimizer()
        task.zero_grad(set_to_none=True)
        train_loader = _to_loader(data.train_pipeline(), "train_pipeline")
        if train_loader is None:
            raise ValueError("fit needs data.train_pipeline() to return a pipeline")
        eval_loader = _to_loader(data.eval_pipeline(), "eval_pipeline")
        if eval_loader is not None:
            _require(task, ("eval_step",), "fit")
        else:
            logger.info("no eval_pipeline; eval skipped")

        accum = cfg.grad_accum_steps
        step, epoch, last_eval_step = 0, 0, None
        window_loss = torch.zeros((), device=self.device)
        log_window: list[torch.Tensor] = []
        train_loss: float | None = None
        eval_metrics: dict[str, float] = {}

        def max_steps_reached() -> bool:
            return cfg.max_steps is not None and step >= cfg.max_steps

        def log_train() -> None:
            nonlocal train_loss
            train_loss = torch.stack(log_window).mean().item()
            log_window.clear()
            lr = opt.param_groups[0]["lr"]
            self._log({"epoch": epoch, "train/loss": train_loss, "lr": lr}, step)

        def run_eval() -> None:
            nonlocal eval_metrics, last_eval_step
            metrics = self._eval_loop(task, eval_loader)
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            self._log({"epoch": epoch, **eval_metrics}, step)
            last_eval_step = step

        def boundary(k: int) -> None:
            nonlocal step, window_loss
            if not torch.isfinite(window_loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            opt.step()
            task.zero_grad(set_to_none=True)
            step += 1
            log_window.append(window_loss / k)
            window_loss = torch.zeros((), device=self.device)
            if step % cfg.log_every_n_steps == 0:
                log_train()
            if (
                eval_loader is not None
                and cfg.eval_every_n_steps
                and step % cfg.eval_every_n_steps == 0
                and not max_steps_reached()
            ):
                run_eval()

        task.train()
        while True:
            n = k = 0
            for batch in train_loader:
                n += 1
                k += 1
                with self._autocast():
                    loss = task.train_step(_to_device(batch, self.device))
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        f"train_step must return a Tensor, got {type(loss).__name__}"
                    )
                if loss.ndim != 0:
                    raise ValueError(
                        f"train_step must return a scalar loss, got shape {tuple(loss.shape)}"
                    )
                scaled = loss / accum
                scaled.backward()  # ponytail: mean of micro-batch means; biased for variable-length batches, sample-weight later.
                window_loss += loss.detach()
                if k == accum:
                    boundary(k)
                    k = 0
                    if max_steps_reached():
                        break
            if max_steps_reached():
                break
            if k:
                boundary(k)
                if max_steps_reached():
                    break
            if n == 0:
                raise ValueError("train epoch yielded no batches")
            epoch += 1
            if cfg.max_epochs and epoch >= cfg.max_epochs:
                break
            if (
                eval_loader is not None
                and cfg.eval_every_n_epochs
                and epoch % cfg.eval_every_n_epochs == 0
                and last_eval_step != step
            ):
                run_eval()

        if log_window:
            log_train()
        if eval_loader is not None and last_eval_step != step:
            run_eval()
        elapsed = time.perf_counter() - start
        logger.info("fit done step=%d epoch=%d elapsed=%.2fs", step, epoch, elapsed)
        return {"step": step, "epoch": epoch, "train/loss": train_loss, **eval_metrics}

    def evaluate(self, task: Task, data: DataFeed) -> dict[str, float]:
        _require(task, ("eval_step",), "evaluate")
        task = self._setup(task)
        loader = _to_loader(data.eval_pipeline(), "eval_pipeline")
        if loader is None:
            raise ValueError("evaluate needs data.eval_pipeline() to return a pipeline")
        metrics = self._eval_loop(task, loader)
        self._log(metrics, None)
        return metrics

    def _setup(self, task: Task) -> Task:
        return task.to(self.device)

    def _log(self, metrics: dict[str, float], step: int | None) -> None:
        items = ([("step", step)] if step is not None else []) + list(metrics.items())
        logger.info(
            " ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in items
            )
        )

    def _eval_loop(self, task, loader) -> dict[str, float]:
        was = task.training
        task.eval()
        try:
            with torch.no_grad():
                sums, counts, n = {}, {}, 0
                for batch in loader:
                    with self._autocast():
                        out = task.eval_step(_to_device(batch, self.device))
                    if not isinstance(out, Mapping):
                        raise TypeError("eval_step must return a dict")
                    n += 1
                    for key, v in out.items():
                        v = torch.as_tensor(v).detach().float()
                        sums[key] = sums.get(key, 0) + v.sum()
                        counts[key] = counts.get(key, 0) + v.numel()
            if n == 0:
                raise ValueError("eval data yielded no batches")
            return {key: (sums[key] / counts[key]).item() for key in sums}
        finally:
            task.train(was)

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=False)
