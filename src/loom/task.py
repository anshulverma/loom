from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler


class Task(nn.Module):
    """A model plus its per-batch logic; the Runner owns the loop.

    Override `train_step` and `build_optimizer` to fit, `eval_step` to evaluate and
    `predict_step` to predict. `forward` is optional.

    Non-tensor state that must survive a resume goes through `get_extra_state` /
    `set_extra_state` (primitives or tensors); a resume-safe counter is a `register_buffer`.
    """

    def train_step(self, batch: Any) -> torch.Tensor:
        """Return the 0-dim mean loss. Never call backward or an optimizer step."""
        raise NotImplementedError

    def eval_step(self, batch: Any) -> dict[str, torch.Tensor]:
        """Return metric values; each is 0-dim or shape [B]."""
        raise NotImplementedError

    def predict_step(self, batch: Any) -> Any:
        """Return any value. A shard reads back with plain `torch.load` only if outputs are
        built from tensors, dict, list, plain tuple and str/int/float/bool/None; the Runner
        never inspects outputs."""
        raise NotImplementedError

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer over `self.parameters()`. Called once per fit, after the
        task moves to its device and before any checkpoint is loaded."""
        raise NotImplementedError

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Optionally return a scheduler, stepped once per optimizer step."""
        return None
