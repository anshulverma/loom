import json

import pytest
import torch
from mlp_classifier import MLPClassifier, make_data

from loom import RunConfig, Runner, seed


def _runner():
    return Runner(RunConfig(device="cpu", max_steps=1))


def test_predict_in_memory():
    seed(0)
    task = MLPClassifier()
    feed = make_data(n_eval=100, batch_size=32)
    outs = _runner().predict(task, feed)
    assert len(outs) == 4 and task.training
    task.eval()
    with torch.no_grad():
        expected = task.predict_step(feed.eval_set.tensors)
    assert torch.allclose(torch.cat(outs), expected)


def test_predict_writes_shards(tmp_path):
    seed(0)
    task = MLPClassifier()
    feed = make_data(n_eval=80, batch_size=16)
    in_memory = _runner().predict(task, feed)
    d = tmp_path / "p"
    paths = _runner().predict(task, feed, output_dir=d, batches_per_shard=2)
    names = [f"part-{i:05d}.pt" for i in range(3)]
    assert paths == [d / n for n in names]
    m = json.loads((d / "manifest.json").read_text())
    assert m == {"shards": names, "num_batches": 5}
    outs = [o for s in m["shards"] for o in torch.load(d / s)]
    assert len(outs) == 5
    assert all(torch.equal(a, b) for a, b in zip(outs, in_memory, strict=True))


def test_predict_nonempty_dir_raises(tmp_path):
    (tmp_path / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        _runner().predict(MLPClassifier(), make_data(n_eval=32), output_dir=tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "keep"
