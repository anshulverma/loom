from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset
from torchdata.nodes import BaseNode

import loom
from loom import DataFeed, RunConfig, Runner, Task, build_pipeline

NUM_FEATURES = 16
NUM_CLASSES = 4
HIDDEN = 32
TEACHER_SEED = 12345


def make_dataset(
    n: int,
    num_features: int = NUM_FEATURES,
    num_classes: int = NUM_CLASSES,
    seed: int = 0,
) -> TensorDataset:
    x = torch.randn(n, num_features, generator=torch.Generator().manual_seed(seed))
    w = torch.randn(
        num_features, num_classes, generator=torch.Generator().manual_seed(TEACHER_SEED)
    )
    return TensorDataset(x, (x @ w).argmax(1))


class SyntheticFeed(DataFeed):
    def __init__(
        self, train_set: TensorDataset, eval_set: TensorDataset, batch_size: int
    ):
        self.train_set = train_set
        self.eval_set = eval_set
        self.batch_size = batch_size

    def train_pipeline(self) -> BaseNode:
        return build_pipeline(self.train_set, batch_size=self.batch_size, shuffle=True)

    def eval_pipeline(self) -> BaseNode:
        return build_pipeline(self.eval_set, batch_size=self.batch_size)

    def predict_pipeline(self) -> BaseNode:
        return build_pipeline(self.eval_set, batch_size=self.batch_size)


def make_data(
    n_train: int = 2048, n_eval: int = 512, batch_size: int = 32, seed: int = 0
) -> SyntheticFeed:
    return SyntheticFeed(
        make_dataset(n_train, seed=seed),
        make_dataset(n_eval, seed=seed + 1),
        batch_size,
    )


class MLPClassifier(Task):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN, NUM_CLASSES),
        )

    def train_step(self, batch):
        x, y = batch
        return F.cross_entropy(self.net(x), y)

    def eval_step(self, batch):
        x, y = batch
        logits = self.net(x)
        return {
            "loss": F.cross_entropy(logits, y, reduction="none"),
            "accuracy": (logits.argmax(1) == y).float(),
        }

    def predict_step(self, batch):
        return self.net(batch[0])

    def build_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-2)


def main(
    out_dir: Path = Path("runs/mlp_classifier"),
    max_epochs: int = 5,
    device: str | None = None,
) -> dict[str, float]:
    """Fit, evaluate and predict; delete runs/mlp_classifier before re-running."""
    loom.seed(0)
    task = MLPClassifier()
    data = make_data()
    runner = Runner(
        RunConfig(
            max_epochs=max_epochs,
            device=device,
            checkpoint_dir=out_dir / "checkpoints",
            checkpoint_every_n_steps=50,
        )
    )
    runner.fit(task, data)
    metrics = runner.evaluate(task, data)
    runner.predict(task, data, output_dir=out_dir / "predictions")
    return metrics


if __name__ == "__main__":
    main()
