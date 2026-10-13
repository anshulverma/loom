# loom

Offline PyTorch loops: train, evaluate, predict, batch inference. Not for serving.

## Quickstart

```bash
uv sync
uv run python examples/mlp_classifier.py  # delete runs/mlp_classifier before re-running
```

```python
from mlp_classifier import MLPClassifier, make_data

from loom import RunConfig, Runner

task, feed, runner = MLPClassifier(), make_data(), Runner(RunConfig(max_epochs=3))
runner.fit(task, feed)
metrics = runner.evaluate(task, feed)
runner.predict(task, feed, output_dir="runs/preds")
```

Resume: `RunConfig(..., checkpoint_dir=..., resume_from="latest")`. A fresh fit refuses a
`checkpoint_dir` holding `step-*.pt` (`FileExistsError`); predict refuses a non-empty `output_dir`.

## Layout

```
src/loom/
    task.py        # Task base class
    data.py        # data pipeline helper
    checkpoint.py  # checkpoint save and resume
    runner.py      # Runner: fit, evaluate, predict
```

## Reading predictions back

```python
import json
from pathlib import Path

import torch

d = Path("runs/mlp_classifier/predictions")
m = json.loads((d / "manifest.json").read_text())
outs = [o for s in m["shards"] for o in torch.load(d / s)]
```

Proprietary; see LICENSE
