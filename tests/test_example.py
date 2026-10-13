from mlp_classifier import main


def test_example_main(tmp_path):
    m = main(tmp_path, max_epochs=3, device="cpu")
    assert m["accuracy"] > 0.9
    assert (
        any((tmp_path / "checkpoints").iterdir())
        and (tmp_path / "predictions" / "manifest.json").exists()
    )
