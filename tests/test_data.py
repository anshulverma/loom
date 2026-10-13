import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchdata.nodes import Loader

from loom.data import _to_loader, build_pipeline


def _loader(n, **kwargs):
    ds = TensorDataset(torch.arange(n))
    return Loader(build_pipeline(ds, **kwargs), restart_on_stop_iteration=False)


def _epoch(loader):
    return [b[0].tolist() for b in loader]


def test_pipeline_yields_every_sample_once():
    loader = _loader(10, batch_size=4)
    assert _epoch(loader) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_resume_mid_epoch_in_epoch_1():
    kwargs = {"batch_size": 3, "shuffle": True, "seed": 7}
    ref = _loader(20, **kwargs)
    epochs = [_epoch(ref) for _ in range(3)]
    assert epochs[1] != epochs[0]

    loader = _loader(20, **kwargs)
    _epoch(loader)
    it = iter(loader)
    for _ in range(2):
        next(it)
    state = loader.state_dict()

    resumed = _loader(20, **kwargs)
    resumed.load_state_dict(state)
    assert _epoch(resumed) == epochs[1][2:]
    assert _epoch(resumed) == epochs[2]


def test_resume_at_epoch_end():
    kwargs = {"batch_size": 3, "shuffle": True, "seed": 7}
    ref = _loader(20, **kwargs)
    epochs = [_epoch(ref) for _ in range(2)]

    loader = _loader(20, **kwargs)
    _epoch(loader)
    state = loader.state_dict()

    resumed = _loader(20, **kwargs)
    resumed.load_state_dict(state)
    assert _epoch(resumed) == []
    assert _epoch(resumed) == epochs[1]


def test_rejects_dataloader_and_loader():
    ds = TensorDataset(torch.arange(4))
    node = build_pipeline(ds, batch_size=2)
    with pytest.raises(TypeError, match="must return torchdata.nodes.BaseNode"):
        _to_loader(DataLoader(ds), "train_pipeline")
    with pytest.raises(TypeError, match="loader.root"):
        _to_loader(Loader(node), "train_pipeline")
    assert _to_loader(None, "eval_pipeline") is None
    assert isinstance(_to_loader(node, "x"), Loader)


def test_drop_last_too_small_raises():
    with pytest.raises(ValueError, match="0 batches per epoch"):
        build_pipeline(TensorDataset(torch.arange(3)), batch_size=4, drop_last=True)
