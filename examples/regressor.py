#!/usr/bin/env -S uv run python
"""Smallest end-to-end loom example: fit, evaluate and predict a linear regressor.

Run from the repo root: ./examples/regressor.py (delete runs/regressor before re-running).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset

import loom
from loom import DataFeed, RunConfig, Runner, Task, build_pipeline


class Regressor(Task):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 1)

    def train_step(self, batch):
        x, y = batch
        return F.mse_loss(self.net(x), y)

    def eval_step(self, batch):
        x, y = batch
        return {"mse": F.mse_loss(self.net(x), y, reduction="none").mean(1)}

    def predict_step(self, batch):
        return self.net(batch[0])

    def build_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-2)


class Feed(DataFeed):
    def __init__(self, dataset):
        self.dataset = dataset

    def train_pipeline(self):
        return build_pipeline(self.dataset, batch_size=32, shuffle=True)

    def eval_pipeline(self):
        return build_pipeline(self.dataset, batch_size=32)

    predict_pipeline = eval_pipeline


def main(
    out_dir: Path = Path("runs/regressor"), device: str | None = None
) -> dict[str, float]:
    loom.seed(0)
    x = torch.randn(256, 8)
    feed = Feed(TensorDataset(x, x.sum(1, keepdim=True)))
    task = Regressor()
    runner = Runner(
        RunConfig(max_epochs=10, device=device, checkpoint_dir=out_dir / "checkpoints")
    )
    runner.fit(task, feed)
    metrics = runner.evaluate(task, feed)
    runner.predict(task, feed, output_dir=out_dir / "predictions")
    return metrics


if __name__ == "__main__":
    print(main())
