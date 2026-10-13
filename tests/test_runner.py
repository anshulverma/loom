from collections import namedtuple

import pytest
import torch
from mlp_classifier import MLPClassifier, make_data
from torch import nn
from torch.utils.data import TensorDataset
from torchdata.nodes import IterableWrapper

from loom import DataFeed, RunConfig, Runner, Task, build_pipeline, seed
from loom.runner import _to_device


def cfg(**kw):
    return RunConfig(device="cpu", **kw)


small = make_data(n_train=256, n_eval=128)


class _ParamTask(Task):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def build_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _Feed(DataFeed):
    def __init__(self, train=None, eval=None):
        self.train, self.eval = train, eval

    def train_pipeline(self):
        return self.train() if self.train else None

    def eval_pipeline(self):
        return self.eval() if self.eval else None


def test_fit_reduces_loss():
    # max_epochs=5: a plain-torch replica measured worst ratio 0.316 over 8 seeds (2 epochs gave 0.674, too tight)
    seed(0)
    task = MLPClassifier()
    r = Runner(cfg(max_epochs=5))
    before = r.evaluate(task, small)["loss"]
    out = r.fit(task, small)
    assert r.evaluate(task, small)["loss"] < 0.5 * before
    assert out["step"] == 40 and out["epoch"] == 5
    assert "train/loss" in out and "eval/accuracy" in out


def test_evaluate_reduction():
    class Probe(_ParamTask):
        def eval_step(self, b):
            return {"per_sample": b[0].float(), "batch_mean": b[0].float().mean()}

    task = Probe()
    before = task.w.detach().clone()
    feed = _Feed(
        eval=lambda: build_pipeline(TensorDataset(torch.arange(10)), batch_size=4)
    )
    m = Runner(cfg(max_steps=1)).evaluate(task, feed)
    assert m == {
        "per_sample": pytest.approx(4.5),
        "batch_mean": pytest.approx((1.5 + 5.5 + 8.5) / 3),
    }
    assert torch.equal(task.w, before)
    assert all(type(v) is float for v in m.values())


def test_device_auto_select():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert Runner(RunConfig(max_steps=1)).device.type == expected


@pytest.mark.parametrize(
    "kw",
    [
        {"max_steps": 1, "grad_accum_steps": 0},
        {"max_steps": 1, "precision": "fp16"},
        {"max_steps": 1, "max_grad_norm": 0.0},
        {},
        {"max_steps": 1, "resume_from": "latest"},
        {"max_steps": 1, "checkpoint_every_n_steps": 5},
    ],
)
def test_invalid_config_raises(kw):
    with pytest.raises(ValueError) as e:
        RunConfig(**kw)
    if kw.get("precision") == "fp16":
        assert str(e.value) == "fp16 needs GradScaler; use bf16"


def test_non_scalar_loss_raises():
    class PerSample(MLPClassifier):
        def train_step(self, batch):
            x, y = batch
            return nn.functional.cross_entropy(self.net(x), y, reduction="none")

    with pytest.raises(ValueError, match="scalar"):
        Runner(cfg(max_steps=1)).fit(PerSample(), small)


def test_non_finite_loss_raises():
    class NanOnThird(MLPClassifier):
        calls = 0

        def train_step(self, batch):
            self.calls += 1
            loss = super().train_step(batch)
            return loss * float("nan") if self.calls == 3 else loss

    with pytest.raises(FloatingPointError, match="step 3"):
        Runner(cfg(max_epochs=1)).fit(NanOnThird(), small)


def test_missing_override_raises():
    class OnlyOptimizer(Task):
        def build_optimizer(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    with pytest.raises(TypeError, match="train_step"):
        Runner(cfg(max_steps=1)).fit(OnlyOptimizer(), small)

    class NoEval(_ParamTask):
        def train_step(self, batch):
            self.calls += 1
            return self.w.sum()

    task = NoEval()
    with pytest.raises(TypeError, match="eval_step"):
        Runner(cfg(max_steps=1)).fit(task, small)
    assert task.calls == 0


def test_to_device_nested():
    NT = namedtuple("NT", "x")
    t = torch.ones(2)
    out = _to_device({"a": [t, "s"], "b": NT(x=t)}, torch.device("meta"))
    assert type(out) is dict and type(out["a"]) is list and type(out["b"]) is NT
    assert out["a"][1] == "s"
    assert out["a"][0].device.type == out["b"].x.device.type == "meta"


def test_fit_pipeline_presence():
    out = Runner(cfg(max_steps=2)).fit(
        MLPClassifier(), _Feed(train=small.train_pipeline)
    )
    assert not any(k.startswith("eval/") for k in out)
    with pytest.raises(ValueError):
        Runner(cfg(max_steps=2)).fit(MLPClassifier(), _Feed(eval=small.eval_pipeline))


def test_empty_train_epoch_raises():
    feed = _Feed(train=lambda: IterableWrapper([]))
    with pytest.raises(ValueError, match="train epoch yielded no batches"):
        Runner(cfg(max_epochs=1)).fit(MLPClassifier(), feed)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a host without CUDA")
def test_unavailable_cuda_raises():
    with pytest.raises(ValueError):
        Runner(RunConfig(max_steps=1, device="cuda"))
