from functools import partial
from typing import Literal

import torch
from torch.utils.data import DistributedSampler, default_collate
from torchdata.nodes import (
    BaseNode,
    Batcher,
    Loader,
    Mapper,
    ParallelMapper,
    Prefetcher,
    SamplerWrapper,
)


class DataFeed:
    """Supplies pipelines. Build them inside these methods (the Runner seeds first). Random
    augmentation is reproducible only with num_workers=0. Bitwise resume
    holds for build_pipeline graphs; a hand-built graph must make order a function of (seed, epoch)."""

    def train_pipeline(self) -> BaseNode | None:
        return None

    def eval_pipeline(self) -> BaseNode | None:
        return None

    def predict_pipeline(self) -> BaseNode | None:
        return None


def build_pipeline(
    dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
    collate_fn=default_collate,
    num_workers: int = 0,
    worker_method: Literal["thread", "process"] = "thread",
    prefetch_factor: int = 2,
) -> BaseNode:
    """Batch a map-style `dataset` (any object with `__len__` and `__getitem__`) into a
    torchdata.nodes graph whose order is a pure function of (seed, epoch).

    `seed=None` resolves to `torch.initial_seed()` at build time, so build pipelines inside
    `DataFeed` methods, after the Runner has seeded.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    if drop_last and len(dataset) < batch_size:
        raise ValueError(
            f"pipeline would yield 0 batches per epoch: drop_last=True, "
            f"len(dataset)={len(dataset)} < batch_size={batch_size}"
        )
    seed = torch.initial_seed() if seed is None else seed
    sampler = DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=shuffle, seed=seed, drop_last=False
    )
    node = Batcher(SamplerWrapper(sampler), batch_size, drop_last=drop_last)
    fetch = partial(_fetch, dataset, collate_fn)
    if num_workers == 0:
        return Mapper(node, fetch)
    node = ParallelMapper(
        node, fetch, num_workers=num_workers, method=worker_method, in_order=True
    )
    return Prefetcher(node, prefetch_factor)


def _fetch(dataset, collate_fn, indices):
    return collate_fn([dataset[i] for i in indices])


def _to_loader(pipeline: object, role: str) -> Loader | None:
    if pipeline is None:
        return None
    if isinstance(pipeline, Loader):
        raise TypeError(
            f"{role}() returned a Loader; return loader.root; the Runner owns the Loader"
        )
    if isinstance(pipeline, BaseNode):
        return Loader(pipeline, restart_on_stop_iteration=False)
    raise TypeError(
        f"{role}() must return torchdata.nodes.BaseNode, got {type(pipeline).__name__}; "
        "use loom.build_pipeline(...) or torchdata.nodes.IterableWrapper"
    )
