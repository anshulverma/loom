# loom

Offline PyTorch model loops: train, evaluate, predict, and offline batch
inference. Not for serving.

## Quickstart

```bash
uv sync
uv run pytest
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
