"""library/speed_gate.py: whether runs made the steps a machine at full speed makes."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("speed_gate", ROOT / "library" / "speed_gate.py")
speed_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed_gate)


def run(steps):
    return {"final": {"num_steps": steps}}


def test_runs_at_speed():
    ok, message = speed_gate.at_speed([run(1258)], 1190)
    assert ok is True
    assert "1,258" in message


def test_a_run_under_the_floor_fails_by_its_steps():
    ok, message = speed_gate.at_speed([run(1258), run(1183)], 1190)
    assert ok is False
    assert "1,183" in message and "1,190" in message
    assert "Keep the machine at one speed" in message


def test_starts_too_far_apart_fail():
    # 1,193 is 5.2% below 1,258: every run is above the floor, but the machine changed speed.
    ok, message = speed_gate.at_speed([run(1258), run(1258), run(1193)], 1100)
    assert ok is False
    assert "1,193" in message and "1,258" in message


def test_starts_within_five_percent_pass():
    # 1,214 is 3.5% below 1,258: the 12 October starts, at full speed.
    assert speed_gate.at_speed([run(1214), run(1258), run(1258)], 1190)[0] is True


def test_cli_exit_status(tmp_path):
    def cli(steps):
        path = tmp_path / f"run-{steps}.json"
        path.write_text(json.dumps(run(steps)), encoding="utf-8")
        return subprocess.run([sys.executable, str(ROOT / "library" / "speed_gate.py"), "1190", str(path)],
                              capture_output=True, text=True)

    fast, slow = cli(1258), cli(1183)
    assert fast.returncode == 0
    assert slow.returncode == 1 and "1,183" in slow.stdout
