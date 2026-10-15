# loom

loom runs offline PyTorch model loops: train, evaluate, predict and offline batch inference. It is not for serving models.

## Install

```bash
pip install loom-torch
```

The package imports as `loom`.

## Quickstart

From a clone of this repo:

```bash
uv sync
uv run pytest
./examples/regressor.py       # smallest example; delete runs/regressor before re-running
./examples/mlp_classifier.py  # adds periodic checkpoints; delete runs/mlp_classifier before re-running
```

## Usage

Subclass `Task` for the model and its step logic. Subclass `DataFeed` for the data pipelines. `Runner` owns the loops.

A `Task` is an `nn.Module` that implements `train_step` (returns the loss) and `build_optimizer`. It adds `eval_step`
(returns a dict of metrics), `predict_step` and `build_scheduler` when it needs them. A `DataFeed` returns pipelines from
`train_pipeline`, `eval_pipeline` and `predict_pipeline`, usually built with `build_pipeline(dataset, batch_size=...)`.
[`examples/regressor.py`](https://github.com/anshulverma/loom/blob/main/examples/regressor.py) shows all of it in about 70 lines.

`RunConfig` holds every run setting: the stop condition (`max_epochs` or `max_steps`), `device`, `checkpoint_dir`, plus
`grad_accum_steps`, `max_grad_norm`, `precision="bf16"`, `checkpoint_every_n_steps` and `eval_every_n_steps`.

To resume a run, set `resume_from="latest"` or a checkpoint path in `RunConfig`. To protect earlier results, a `fit` that
does not resume raises `FileExistsError` when `checkpoint_dir` already holds `step-*.pt` files. `predict` raises
`FileExistsError` when `output_dir` is non-empty.

## Layout

```
src/loom/
    task.py        # Task base class
    data.py        # DataFeed and build_pipeline (torchdata.nodes)
    checkpoint.py  # checkpoint save and resume
    runner.py      # RunConfig and Runner: fit, evaluate, predict
examples/
    regressor.py       # smallest end-to-end example
    mlp_classifier.py  # MLP classifier with periodic checkpoints
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

MIT. See LICENSE.
