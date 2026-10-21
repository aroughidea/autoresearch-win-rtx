"""Did these runs train at full speed? Every run gets the same training time, so a slow machine makes
fewer steps and worse scores with no change to the recipe (README: "Keep the machine at one speed").

    uv run library/speed_gate.py <floor> <run.json>...

Exits 0 when every run made at least <floor> steps and the runs are within 5% of each other, else 1.
On this laptop at 600 s, TinyStories with its own vocabulary: recipe A makes about 1,258 steps (floor
1,190), recipe B about 3,137 (floor 2,980).
"""
import json
import sys
from pathlib import Path

ADVICE = 'Stop, fix the machine (README: "Keep the machine at one speed"), and run again.'


def at_speed(runs, floor, spread=0.05):
    """(True, message) when every run made at least `floor` steps and the fewest are within `spread`
    of the most; (False, message) naming the steps otherwise."""
    steps = [run["final"]["num_steps"] for run in runs]
    low, high = min(steps), max(steps)
    listed = ", ".join(f"{s:,}" for s in steps)
    if low < floor:
        return False, f"{listed} steps: {low:,} is under the floor of {floor:,}, so the machine ran slow. {ADVICE}"
    if high - low > spread * high:
        return False, (f"{listed} steps: {low:,} is {(high - low) / high:.1%} below {high:,}, more than "
                       f"{spread:.0%}, so the machine changed speed. {ADVICE}")
    return True, f"{listed} steps: at speed (floor {floor:,})."


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or not argv[0].isdigit():
        print("usage: uv run library/speed_gate.py <floor> <run.json>...")
        return 2
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in argv[1:]]
    ok, message = at_speed(runs, int(argv[0]))
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
