# Contributing

## Setup

```bash
uv sync
```

## Before each commit

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

## Conventions

- One logical change per commit, with an imperative subject.
- snake_case files, PascalCase classes.
- A test with every feature.
- GPU tests go behind `skipif(not torch.cuda.is_available())`.
- Never commit checkpoints or predictions.
- Split a module only past about 400 lines.
- Do not add: conftest, `py.typed`, a utils module, a type checker, CI,
  pre-commit, or a Makefile.

## Banned identifiers

These names must not appear anywhere in the code, tests, or docs (this list is
the one exception):

- LightningModule
- Trainer
- Unit
- AutoUnit
- training_step
- configure_optimizers
- DataModule
- State
- seed_everything
- train_dataloader
- val_dataloader
- predict_dataloader
- validation_step
- global_step
- Callback
- TrainUnit
- EvalUnit
- PredictUnit
- ModelCheckpoint
