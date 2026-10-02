import json

from velib.cli import main


def test_demo_runs_the_whole_pipeline(tmp_path):
    exit_code = main(
        [
            "-q",
            "--data-dir",
            str(tmp_path),
            "demo",
            "--stations",
            "10",
            "--days",
            "22",
            "--test-days",
            "3",
        ]
    )
    assert exit_code == 0
    for relative in [
        "raw",
        "processed/hourly.pkl",
        "models/model.joblib",
        "models/metrics.json",
        "models/diagnostics.json",
        "4_compressed_predictions_final_fix.json",
    ]:
        assert (tmp_path / relative).exists(), relative

    doc = json.loads((tmp_path / "forecast.json").read_text())
    assert doc["synthetic"] is True
    assert len(doc["stations"]) == 9  # one synthetic station stops reporting halfway

    metrics = json.loads((tmp_path / "models" / "metrics.json").read_text())
    assert "model" not in metrics  # the estimator itself stays in model.joblib
    assert set(metrics["metrics"]["scores"]) == {"model", "seasonal_naive", "weekly_profile"}


def test_steps_report_missing_inputs_instead_of_crashing(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "prepare"]) == 1
    assert "velib fetch" in capsys.readouterr().err
    assert main(["--data-dir", str(tmp_path), "train"]) == 1
    assert "velib prepare" in capsys.readouterr().err
