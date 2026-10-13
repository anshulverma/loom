import shutil

import pytest
import torch
from mlp_classifier import MLPClassifier, SyntheticFeed, make_dataset

from loom import (
    DataFeed,
    RunConfig,
    Runner,
    build_pipeline,
    latest_checkpoint,
    load_checkpoint,
    seed,
)


class _SchedMLP(MLPClassifier):
    def build_scheduler(self, opt):
        return torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)


_data = SyntheticFeed(make_dataset(40), make_dataset(16, seed=1), batch_size=6)
_base = {
    "device": "cpu",
    "grad_accum_steps": 2,
    "max_grad_norm": 1.0,
    "precision": "bf16",
    "max_epochs": 3,
    "checkpoint_every_n_steps": 1,
}


def _fit(**kw):
    seed(0)
    return Runner(RunConfig(**{**_base, **kw})).fit(_SchedMLP(), _data)


def _assert_same_state(a, b):
    a, b = load_checkpoint(a), load_checkpoint(b)
    assert (a["step"], a["epoch"]) == (b["step"], b["epoch"])
    assert a["task"].keys() == b["task"].keys()
    for key in a["task"]:
        assert torch.equal(a["task"][key], b["task"][key]), key
    sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
    assert sa.keys() == sb.keys()
    for i in sa:
        assert sa[i].keys() == sb[i].keys()
        for name in sa[i]:
            assert torch.equal(sa[i][name], sb[i][name]), (i, name)


def test_resume_is_bitwise_identical(tmp_path):
    # 7 batches/epoch, N=2 -> 4 steps/epoch: step 2 mid-epoch, step 4 the epoch-end partial window,
    # step 7 mid-epoch 1
    ref = tmp_path / "ref"
    assert _fit(checkpoint_dir=ref)["step"] == 12
    final = "step-00000012.pt"
    for s in (2, 4, 7):
        d = tmp_path / f"d{s}"
        d.mkdir()
        shutil.copy(ref / f"step-{s:08d}.pt", d)
        _fit(checkpoint_dir=d, resume_from=d / f"step-{s:08d}.pt")
        _assert_same_state(ref / final, d / final)

    ext = tmp_path / "ext"
    _fit(checkpoint_dir=ext, max_epochs=1)
    _fit(checkpoint_dir=ext, resume_from="latest")
    _assert_same_state(ref / final, ext / final)


def test_failed_save_keeps_previous(tmp_path, monkeypatch):
    real_save, calls = torch.save, []

    def flaky_save(obj, f):
        calls.append(1)
        if len(calls) == 2:
            f.write(b"partial")
            raise OSError("disk full")
        real_save(obj, f)

    monkeypatch.setattr("loom.checkpoint.torch.save", flaky_save)
    with pytest.raises(OSError, match="disk full"):
        _fit(checkpoint_dir=tmp_path, max_steps=3)
    assert load_checkpoint(tmp_path / "step-00000001.pt")["step"] == 1
    assert not (tmp_path / "step-00000002.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_latest_and_dir_guard(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _fit(checkpoint_dir=empty, max_steps=3, resume_from="latest")["step"] == 3

    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "notes.txt").write_text("x")
    assert _fit(checkpoint_dir=stray, max_steps=1)["step"] == 1

    with pytest.raises(FileExistsError):
        _fit(checkpoint_dir=empty, max_steps=3)
    with pytest.raises(FileExistsError):
        _fit(checkpoint_dir=empty, max_steps=3, resume_from=empty / "step-00000002.pt")


def test_missing_resume_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _fit(checkpoint_dir=tmp_path, resume_from=tmp_path / "nope.pt")


def test_resume_after_completed_fit_returns(tmp_path):
    kw = {"checkpoint_dir": tmp_path, "max_epochs": 1, "checkpoint_every_n_steps": None}
    s = _fit(**kw)["step"]
    files = sorted(p.name for p in tmp_path.iterdir())
    assert _fit(**kw, resume_from="latest") == {"step": s, "epoch": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == files


def test_resume_changed_pipeline_structure_raises(tmp_path):
    class Feed(DataFeed):
        def __init__(self, workers):
            self.workers = workers

        def train_pipeline(self):
            return build_pipeline(
                make_dataset(64), batch_size=8, shuffle=True, num_workers=self.workers
            )

    def fit(workers, **kw):
        cfg = RunConfig(device="cpu", checkpoint_dir=tmp_path, **kw)
        return Runner(cfg).fit(MLPClassifier(), Feed(workers))

    fit(0, max_steps=2, checkpoint_every_n_steps=1)
    with pytest.raises(RuntimeError, match="train pipeline structure differs"):
        fit(2, max_steps=4, resume_from="latest")
    assert latest_checkpoint(tmp_path).name == "step-00000002.pt"


def test_resume_at_epoch_boundary_returns_only_floats(tmp_path):
    data = SyntheticFeed(make_dataset(64), make_dataset(16, seed=1), batch_size=8)
    cfg = {"device": "cpu", "checkpoint_dir": tmp_path}
    Runner(RunConfig(**cfg, max_steps=8)).fit(MLPClassifier(), data)
    out = Runner(RunConfig(**cfg, max_epochs=1, resume_from="latest")).fit(
        MLPClassifier(), data
    )
    assert (out["step"], out["epoch"]) == (8, 1)
    assert "train/loss" not in out
    assert all(isinstance(v, (int, float)) for v in out.values())
