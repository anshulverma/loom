# loom

Offline PyTorch model loops: train, evaluate, predict, and offline batch
inference. Not for serving.

## Quickstart

```bash
uv sync
uv run pytest
uv run python examples/mlp_classifier.py  # delete runs/mlp_classifier before re-running
```

```python
from loom import DataFeed, RunConfig, Runner, Task, build_pipeline


class Net(Task): ...  # train_step, eval_step, predict_step, build_optimizer


class Feed(DataFeed): ...  # *_pipeline() returns build_pipeline(ds, batch_size=32)


task, feed, runner = Net(), Feed(), Runner(RunConfig(max_epochs=3))
runner.fit(task, feed)
metrics = runner.evaluate(task, feed)
runner.predict(task, feed, output_dir="runs/preds")
```

## Layout

```
src/loom/
    task.py        # Task base class
    data.py        # data pipeline helper
    checkpoint.py  # checkpoint save and resume
    runner.py      # Runner: fit, evaluate, predict
examples/
tests/
```

## Reading predictions back

```python
m = json.loads((d / "manifest.json").read_text())
outs = [o for s in m["shards"] for o in torch.load(d / s)]
```

Proprietary; see LICENSE
