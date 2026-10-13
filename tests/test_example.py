import os

from mlp_classifier import main


def test_example_main(tmp_path):
    m = main(tmp_path, max_epochs=3, device="cpu")
    assert m["accuracy"] > 0.9
    assert (
        any((tmp_path / "checkpoints").iterdir())
        and (tmp_path / "predictions" / "manifest.json").exists()
    )


def test_regressor_example_is_executable(tmp_path):
    from pathlib import Path

    import regressor

    examples = Path(regressor.__file__).parent
    for script in (examples / "regressor.py", examples / "mlp_classifier.py"):
        assert script.read_text().startswith("#!/usr/bin/env -S uv run python\n")
        assert os.access(script, os.X_OK)
    metrics = regressor.main(tmp_path, device="cpu")
    assert set(metrics) == {"mse"} and metrics["mse"] < 0.1
    assert (tmp_path / "predictions" / "manifest.json").exists()
