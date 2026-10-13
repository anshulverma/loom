# loom

Offline PyTorch model loops: train, evaluate, predict, and offline batch
inference. Not for serving.

## Quickstart

```bash
uv sync
uv run pytest
uv run python examples/mlp_classifier.py  # delete runs/mlp_classifier before re-running
```

## Usage

Subclass `Task` (model + step logic) and `DataFeed` (pipelines); `Runner` owns the loops.

```python
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset

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
        return torch.optim.AdamW(self.parameters(), lr=1e-2)


class Feed(DataFeed):
    def __init__(self, dataset):
        self.dataset = dataset

    def train_pipeline(self):
        return build_pipeline(self.dataset, batch_size=32, shuffle=True)

    def eval_pipeline(self):
        return build_pipeline(self.dataset, batch_size=32)

    predict_pipeline = eval_pipeline


x = torch.randn(256, 8)
feed = Feed(TensorDataset(x, x.sum(1, keepdim=True)))
task = Regressor()
runner = Runner(RunConfig(max_epochs=3, checkpoint_dir="runs/regressor/checkpoints"))
runner.fit(task, feed)
print(runner.evaluate(task, feed))
runner.predict(task, feed, output_dir="runs/regressor/predictions")
```

`RunConfig` also takes `grad_accum_steps`, `max_grad_norm`, `precision="bf16"`,
`checkpoint_every_n_steps` and `eval_every_n_steps`.

Resume with `resume_from="latest"` (or a checkpoint path). `FileExistsError` guards data: a fresh
`fit` refuses a `checkpoint_dir` holding `step-*.pt`; `predict` refuses a non-empty `output_dir`.

## Layout

```
src/loom/
    task.py        # Task base class
    data.py        # DataFeed and build_pipeline (torchdata.nodes)
    checkpoint.py  # checkpoint save and resume
    runner.py      # RunConfig and Runner: fit, evaluate, predict
examples/          # end-to-end MLP classifier
tests/
```

## Reading predictions back

```python
import json
from pathlib import Path

import torch

d = Path("runs/regressor/predictions")
m = json.loads((d / "manifest.json").read_text())
outs = [o for s in m["shards"] for o in torch.load(d / s)]
```

Proprietary; see LICENSE
