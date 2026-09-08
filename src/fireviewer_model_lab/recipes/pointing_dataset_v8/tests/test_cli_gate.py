import json
import subprocess
import sys


def test_require_ready_exits_two_for_empty_draft(tmp_path):
    manifest, baseline, history, output = [tmp_path / name for name in ("rows.jsonl", "baseline.jsonl", "history.json", "report.json")]
    manifest.write_text("")
    baseline.write_text("")
    history.write_text(json.dumps({"sha256": {}, "source_groups": {}}))
    result = subprocess.run([sys.executable, "-m", "training.pointing_dataset_v8.audit_coverage", "--manifest", str(manifest),
                             "--baseline", str(baseline), "--history", str(history), "--output", str(output), "--require-ready"],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 2, result.stderr
    assert json.loads(output.read_text())["ready"] is False
