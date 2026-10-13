import json
import logging
import os
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from loom.checkpoint import (
    _checkpoint_steps,
    _jsonable_config,
    atomic_save,
    latest_checkpoint,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
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


def _resolve_resume(cfg: RunConfig) -> dict | None:
    path = cfg.resume_from
    if path == "latest":
        path = latest_checkpoint(cfg.checkpoint_dir)
    elif path is not None and not Path(path).is_file():
        raise FileNotFoundError(f"resume_from {str(path)!r} is not a file")
    ckpt = None if path is None else load_checkpoint(path)
    if cfg.checkpoint_dir is not None:
        newest = max(_checkpoint_steps(cfg.checkpoint_dir), default=None)
        if newest is not None and ckpt is None:
            raise FileExistsError(
                f"{cfg.checkpoint_dir} holds checkpoints up to step {newest}; "
                "resume with resume_from='latest' or use an empty checkpoint_dir"
            )
        if newest is not None and newest > ckpt["step"]:
            raise FileExistsError(
                f"{cfg.checkpoint_dir} holds step {newest}, newer than the "
                f"resume point step {ckpt['step']}"
            )
    return ckpt


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
        ckpt = _resolve_resume(cfg)
        seed(cfg.seed)
        _require(task, ("train_step", "build_optimizer"), "fit")
        task = self._setup(task)
        opt = task.build_optimizer()
        sched = task.build_scheduler(opt)
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise TypeError(
                "ReduceLROnPlateau needs a metric; use a step-based scheduler"
            )
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
        step, epoch, in_epoch, last_eval_step = 0, 0, False, None
        wrap_iter = ckpt is not None
        window_loss = torch.zeros((), device=self.device)
        log_window: list[torch.Tensor] = []
        train_loss: float | None = None
        last_norm = torch.zeros((), device=self.device)
        eval_metrics: dict[str, float] = {}

        def max_steps_reached() -> bool:
            return cfg.max_steps is not None and step >= cfg.max_steps

        def stop_reached() -> bool:
            return max_steps_reached() or (
                cfg.max_epochs is not None and epoch >= cfg.max_epochs
            )

        def epoch_iter():
            nonlocal wrap_iter
            if not wrap_iter:
                return iter(train_loader)
            wrap_iter = False
            try:
                return iter(train_loader)
            except KeyError as err:
                raise RuntimeError(
                    "train pipeline structure differs from checkpoint "
                    "(num_workers 0<->N or prefetch changed?)"
                ) from err

        def save() -> None:
            save_checkpoint(
                Path(cfg.checkpoint_dir),
                step=step,
                epoch=epoch,
                in_epoch=in_epoch,
                task=task,
                optimizer=opt,
                scheduler=sched,
                train_loader=train_loader,
                config=cfg,
                device=self.device,
            )

        def log_train() -> None:
            nonlocal train_loss
            train_loss = torch.stack(log_window).mean().item()
            log_window.clear()
            lr = opt.param_groups[0]["lr"]
            metrics = {"epoch": epoch, "train/loss": train_loss, "lr": lr}
            if cfg.max_grad_norm is not None:
                metrics["grad_norm"] = last_norm.item()
            self._log(metrics, step)

        def run_eval() -> None:
            nonlocal eval_metrics, last_eval_step
            metrics = self._eval_loop(task, eval_loader)
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            self._log({"epoch": epoch, **eval_metrics}, step)
            last_eval_step = step

        def boundary(k: int) -> None:
            nonlocal step, window_loss, last_norm
            if not torch.isfinite(window_loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            if cfg.max_grad_norm is not None:
                last_norm = torch.nn.utils.clip_grad_norm_(
                    task.parameters(), cfg.max_grad_norm
                )
                if not torch.isfinite(last_norm):
                    raise FloatingPointError(f"non-finite grad norm at step {step + 1}")
            opt.step()
            if sched is not None:
                sched.step()
            task.zero_grad(set_to_none=True)
            step += 1
            log_window.append(window_loss / k)
            window_loss = torch.zeros((), device=self.device)
            if step % cfg.log_every_n_steps == 0:
                log_train()
            if max_steps_reached():
                return
            if (
                eval_loader is not None
                and cfg.eval_every_n_steps
                and step % cfg.eval_every_n_steps == 0
            ):
                run_eval()
            if (
                cfg.checkpoint_every_n_steps
                and step % cfg.checkpoint_every_n_steps == 0
            ):
                save()

        if ckpt is not None:
            if (ckpt["scheduler"] is None) != (sched is None):
                raise ValueError("build_scheduler presence differs from checkpoint")
            task.load_state_dict(ckpt["task"], strict=True)
            opt.load_state_dict(ckpt["optimizer"])
            if sched is not None:
                sched.load_state_dict(ckpt["scheduler"])
            train_loader.load_state_dict(ckpt["train_loader"])
            step, epoch, in_epoch = ckpt["step"], ckpt["epoch"], ckpt["in_epoch"]
            old = ckpt["config"]
            for key, new in _jsonable_config(cfg).items():
                if key != "resume_from" and old.get(key) != new:
                    logger.warning("%s: %s -> %s", key, old.get(key), new)
            if cfg.eval_every_n_steps and step % cfg.eval_every_n_steps == 0:
                last_eval_step = step
            if not in_epoch and next(epoch_iter(), None) is not None:
                raise RuntimeError("closed epoch had batches left")
            restore_rng(ckpt["rng"], self.device)
        if stop_reached():
            logger.info(
                "resumed at step=%d epoch=%d, already at a stop condition", step, epoch
            )
            return {"step": step, "epoch": epoch}

        task.train()
        while True:
            resumed_tail = in_epoch
            in_epoch, n, k = True, 0, 0
            for batch in epoch_iter():
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
                for p in task.parameters():
                    if p.grad is not None:
                        p.grad.mul_(accum / k)
                boundary(k)
                if max_steps_reached():
                    break
            if n == 0 and not resumed_tail:
                raise ValueError("train epoch yielded no batches")
            epoch += 1
            in_epoch = False
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
        if cfg.checkpoint_dir is not None:
            save()
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

    def predict(
        self,
        task: Task,
        data: DataFeed,
        *,
        output_dir: str | os.PathLike[str] | None = None,
        batches_per_shard: int = 100,
    ) -> list[Any] | list[Path]:
        """Per-batch CPU outputs in memory, or shard paths when `output_dir` is set."""
        if batches_per_shard < 1:
            raise ValueError(f"batches_per_shard must be >= 1, got {batches_per_shard}")
        _require(task, ("predict_step",), "predict")
        out_dir = None if output_dir is None else Path(output_dir)
        if out_dir is not None:
            if out_dir.exists() and any(out_dir.iterdir()):
                raise FileExistsError(f"{out_dir} is not empty")
            out_dir.mkdir(parents=True, exist_ok=True)
        loader = _to_loader(data.predict_pipeline(), "predict_pipeline")
        if loader is None:
            raise ValueError(
                "predict needs data.predict_pipeline() to return a pipeline"
            )
        task = self._setup(task)
        cpu = torch.device("cpu")
        results: list[Any] = []
        shards: list[Path] = []
        n = 0
        was = task.training
        task.eval()
        try:
            # ponytail: may move to inference_mode if profiling justifies it.
            with torch.no_grad():
                for batch in loader:
                    with self._autocast():
                        out = task.predict_step(_to_device(batch, self.device))
                    results.append(_to_device(out, cpu))
                    n += 1
                    if out_dir is not None and len(results) == batches_per_shard:
                        shards.append(self._write_shard(out_dir, len(shards), results))
                        results = []
        finally:
            task.train(was)
        if out_dir is None:
            return results
        if results:
            shards.append(self._write_shard(out_dir, len(shards), results))
        manifest = {"shards": [p.name for p in shards], "num_batches": n}
        atomic_save(json.dumps(manifest).encode(), out_dir / "manifest.json")
        logger.info(
            "predicted %d batches into %d shards in %s", n, len(shards), out_dir
        )
        return shards

    def _write_shard(self, out_dir: Path, i: int, outputs: list[Any]) -> Path:
        path = out_dir / f"part-{i:05d}.pt"
        atomic_save(outputs, path)
        logger.info("wrote %s (%d batches)", path, len(outputs))
        return path

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
        return torch.autocast(
            self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bf16",
        )
