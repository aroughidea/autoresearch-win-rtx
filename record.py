"""
The experiment record: one folder per session (sessions/<name>/) holding every version of
train.py that trained (versions/), one entry per run (runs/), and a pointer to the best version
(best.json). lab.py is its command line; capture.py calls begin_run() as each training run starts.

Standard library only, so lab.py starts fast. The agent must not modify this file (see program.md).
"""

import contextlib
import datetime as _dt
import difflib
import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

SESSIONS = Path("sessions")
FIXED_FILES = ("prepare.py", "capture.py", "record.py", "lab.py", "program.md", "pyproject.toml", "uv.lock")
RESULTS_HEADER = ("timestamp", "commit", "val_bpb", "memory_gb", "status", "description")


class RecordError(RuntimeError):
    """Something the record refuses, said plainly with what to do instead."""


def compact_timestamp(moment):
    """2026-05-23 15:57:43 -07:00 -> '20260523T155743-0700'. Colon-free, safe in Windows filenames."""
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.strftime("%Y%m%dT%H%M%S%z")


def version_id(text):
    """The first 12 hex characters of the SHA-256 of the code, with line endings made uniform."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:12]


def write_atomic(path, text):
    """Write to a temporary file, then rename: a reader sees the old file or the new one, never half."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data):
    write_atomic(path, json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Session:
    """One session's folder."""

    def __init__(self, root):
        self.root = Path(root)
        self.meta = _read_json(self.root / "session.json")

    @property
    def name(self):
        return self.meta["name"]

    @property
    def versions_dir(self):
        return self.root / "versions"

    @property
    def runs_dir(self):
        return self.root / "runs"

    @property
    def best_path(self):
        return self.root / "best.json"

    @property
    def current_path(self):
        return self.root / "current.json"

    @property
    def results_path(self):
        return self.root / "results.tsv"


def start_session(name, *, dataset, tokenizer, run_minutes=5, hours=8, recipe_path="train.py",
                  base=SESSIONS, project_root=".", now=None):
    """Create sessions/<name>/, store train.py as the first version, point best.json at it with no
    score, record the fixed files' hashes, and make it the active session."""
    root = Path(base) / name
    if root.exists():
        raise RecordError(f"record: session {name} already exists ({root})")
    now = now or _dt.datetime.now().astimezone()
    project_root = Path(project_root)
    meta = {
        "name": name, "dataset": dataset, "tokenizer": tokenizer,
        "run_minutes": int(run_minutes), "hours": float(hours),
        "started": now.isoformat(timespec="seconds"),
        "ends": (now + _dt.timedelta(hours=float(hours))).isoformat(timespec="seconds"),
        "fixed": {f: _file_sha256(project_root / f) for f in FIXED_FILES if (project_root / f).exists()},
    }
    _write_json(root / "session.json", meta)
    session = Session(root)
    first = store_version(session, Path(recipe_path).read_text(encoding="utf-8"))
    _write_json(session.best_path, {"version": first, "run": None, "val_bpb": None})
    write_results_tsv(session)
    write_atomic(Path(base) / "active.txt", name + "\n")
    return session


def active_session(base=SESSIONS):
    """The session named in sessions/active.txt, or None when there is none. A name whose folder is
    missing is an error: falling back silently would send runs to the wrong place."""
    try:
        name = (Path(base) / "active.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not name:
        return None
    root = Path(base) / name
    if not (root / "session.json").exists():
        raise RecordError(f"record: sessions/active.txt names {name}, but {root} has no session.json")
    return Session(root)


def store_version(session, text):
    """Save this code in versions/ (once per distinct code) and return its id."""
    vid = version_id(text)
    path = session.versions_dir / f"{vid}.py"
    if not path.exists():
        write_atomic(path, text.replace("\r\n", "\n"))
    return vid


def read_version(session, vid):
    path = session.versions_dir / f"{vid}.py"
    if not path.exists():
        raise RecordError(f"record: no version {vid} in {session.root}")
    return path.read_text(encoding="utf-8")


def read_best(session):
    return _read_json(session.best_path)


def changed_fixed_files(session, project_root="."):
    """The fixed files whose content differs from the session's start (or that went missing)."""
    changed = []
    for name, digest in session.meta.get("fixed", {}).items():
        path = Path(project_root) / name
        if not path.exists() or _file_sha256(path) != digest:
            changed.append(name)
    return changed


def entries(session):
    """Every run entry, oldest first (run ids start with the time)."""
    if not session.runs_dir.exists():
        return []
    return [_read_json(p) for p in sorted(session.runs_dir.glob("*.json"))]


def undecided(session):
    """Finished runs that have no keep or discard yet."""
    return [e for e in entries(session) if e.get("status") is None]


def unfinished(session):
    """The run that started but wrote no entry (still training, or crashed), else None."""
    try:
        current = _read_json(session.current_path)
    except (OSError, ValueError):
        return None
    return None if (session.runs_dir / f"{current['run_id']}.json").exists() else current


def _crash_entry(current, description, stamp):
    return {"schema": 1, "run_id": current["run_id"], "commit": current["version"],
            "version": current["version"], "parent": current["parent"], "created": current["started"],
            "status": "crash", "description": description, "decided": stamp,
            "final": {"val_bpb": None, "peak_vram_mb": None}}


def _now(now):
    return (now or _dt.datetime.now().astimezone()).isoformat(timespec="seconds")


def begin_run(session, *, dataset, tokenizer, time_budget_s, recipe_path="train.py", project_root=".", now=None):
    """Called by capture.py as a training run starts: check the run belongs in this session, store
    the code about to train, and note the run in current.json. Refuses, with the reason, otherwise."""
    changed = changed_fixed_files(session, project_root)
    if changed:
        raise RecordError(f"record: {', '.join(changed)} changed since the session started; restore "
                          "before training (only train.py may change)")
    if (dataset, tokenizer) != (session.meta["dataset"], session.meta["tokenizer"]):
        raise RecordError(f"record: this session is {session.meta['dataset']}/{session.meta['tokenizer']}, "
                          f"but {dataset}/{tokenizer} is active; prepare the session's pair first")
    if int(time_budget_s) != session.meta["run_minutes"] * 60:
        raise RecordError(f"record: this session's runs train {session.meta['run_minutes']} min, but this "
                          f"run would train {int(time_budget_s)} s; check AUTORESEARCH_TIME_BUDGET")
    gone = unfinished(session)
    if gone is not None:
        _write_json(session.runs_dir / f"{gone['run_id']}.json", _crash_entry(gone, "did not finish", _now(now)))
    text = Path(recipe_path).read_text(encoding="utf-8")
    vid = version_id(text)
    waiting = {e["version"] for e in undecided(session)}
    if waiting and waiting != {vid}:
        raise RecordError('record: the last run has no decision yet; run `uv run lab.py keep "..."` or '
                          '`uv run lab.py undo "..."` before training a different train.py')
    store_version(session, text)
    moment = now or _dt.datetime.now().astimezone()
    current = {"run_id": f"{compact_timestamp(moment)}_{vid}", "version": vid,
               "parent": read_best(session)["version"], "started": moment.isoformat(timespec="seconds")}
    _write_json(session.current_path, current)
    return current


def decide(session, keep, description, *, recipe_path="train.py", checkpoint="checkpoint_pre_eval.pt",
           checkpoints_dir="checkpoints", now=None):
    """The agent's decision on the runs since the last one. Keep: the pointer moves to their version
    and the model is archived. Undo: they are discarded (or a run that never finished is a crash)
    and train.py is reset from the best version. best.json is written last."""
    description = " ".join((description or "").split())
    if not description:
        raise RecordError("record: say in a few words what this experiment tried")
    stamp = _now(now)
    gone = unfinished(session)
    if gone is not None:
        if keep:
            raise RecordError("record: the last run did not finish, so it can only be undone")
        _write_json(session.runs_dir / f"{gone['run_id']}.json", _crash_entry(gone, description, stamp))
        write_atomic(recipe_path, read_version(session, read_best(session)["version"]))
        write_results_tsv(session)
        return {"status": "crash", "runs": [gone["run_id"]], "best": read_best(session)}
    targets = undecided(session)
    if not targets:
        raise RecordError("record: no run has finished since the last decision")
    status = "keep" if keep else "discard"
    best = None
    if keep:
        last = targets[-1]
        if Path(checkpoint).exists():
            archived = Path(checkpoints_dir) / f"{last['run_id']}.pt"
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, archived)
            last["checkpoint"] = archived.as_posix()
            last["checkpoint_sha256"] = _file_sha256(archived)
        scores = [e["final"]["val_bpb"] for e in targets if (e.get("final") or {}).get("val_bpb") is not None]
        best = {"version": last["version"], "run": last["run_id"],
                "val_bpb": round(sum(scores) / len(scores), 6) if scores else None}
    for e in targets:
        e["status"], e["description"], e["decided"] = status, description, stamp
        _write_json(session.runs_dir / f"{e['run_id']}.json", e)
    if keep:
        _write_json(session.best_path, best)
    else:
        write_atomic(recipe_path, read_version(session, read_best(session)["version"]))
    write_results_tsv(session)
    return {"status": status, "runs": [e["run_id"] for e in targets], "best": read_best(session)}


def history(session):
    """One row per run, oldest first: what lab.py history prints and results.tsv holds."""
    best = read_best(session)
    rows = []
    for e in entries(session):
        final = e.get("final") or {}
        vram = final.get("peak_vram_mb")
        rows.append({"run": e["run_id"], "created": e.get("created") or "", "version": e.get("version") or "",
                     "parent": e.get("parent") or "", "val_bpb": final.get("val_bpb"),
                     "memory_gb": round(vram / 1024, 1) if vram else None,
                     "status": e.get("status") or "", "description": e.get("description") or "",
                     "best": e["run_id"] == best.get("run")})
    return rows


def _results_text(session):
    lines = ["\t".join(RESULTS_HEADER)]
    for r in history(session):
        if not r["status"]:
            continue  # an undecided run joins the scoreboard when the agent decides
        lines.append("\t".join([
            r["created"], r["version"],
            f"{r['val_bpb']:.6f}" if r["val_bpb"] is not None else "0.000000",
            f"{r['memory_gb']:.1f}" if r["memory_gb"] is not None else "0.0",
            r["status"], " ".join(r["description"].split()),
        ]))
    return "\n".join(lines) + "\n"


def write_results_tsv(session):
    """results.tsv, generated from the entries; never edited by hand."""
    write_atomic(session.results_path, _results_text(session))


def diff(session, a=None, b=None, recipe_path="train.py"):
    """A unified diff between two versions. 'best' and 'train.py' name the pointer's version and the
    working copy; by default, the best version against train.py."""
    def load(ref):
        if ref in (None, "train.py"):
            return Path(recipe_path).read_text(encoding="utf-8").replace("\r\n", "\n"), "train.py"
        vid = read_best(session)["version"] if ref == "best" else ref
        return read_version(session, vid), vid
    left, left_name = load(a or "best")
    right, right_name = load(b)
    return "".join(difflib.unified_diff(left.splitlines(keepends=True), right.splitlines(keepends=True),
                                        left_name, right_name))


def export_sqlite(session, path):
    """A SQLite copy of the record, for searching: tables versions, runs and best."""
    path = Path(path)
    path.unlink(missing_ok=True)
    with contextlib.closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE versions (id TEXT PRIMARY KEY, code TEXT)")
        db.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, version TEXT, parent TEXT, created TEXT, "
                   "val_bpb REAL, memory_gb REAL, status TEXT, description TEXT, entry TEXT)")
        db.execute("CREATE TABLE best (version TEXT, run TEXT, val_bpb REAL)")
        for p in sorted(session.versions_dir.glob("*.py")):
            db.execute("INSERT INTO versions VALUES (?, ?)", (p.stem, p.read_text(encoding="utf-8")))
        for e, r in zip(entries(session), history(session)):
            db.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (r["run"], r["version"], r["parent"], r["created"], r["val_bpb"], r["memory_gb"],
                        r["status"], r["description"], json.dumps(e, ensure_ascii=False)))
        b = read_best(session)
        db.execute("INSERT INTO best VALUES (?, ?, ?)", (b["version"], b["run"], b["val_bpb"]))
        db.commit()
    return path


def check(session, project_root="."):
    """What is inconsistent in the record, as sentences; an empty list means nothing is."""
    problems = [f"{name} changed since the session started" for name in changed_fixed_files(session, project_root)]
    for e in entries(session):
        if not (session.versions_dir / f"{e.get('version')}.py").exists():
            problems.append(f"{e['run_id']}: its version {e.get('version')} is missing")
    best = read_best(session)
    if not (session.versions_dir / f"{best['version']}.py").exists():
        problems.append(f"best.json names version {best['version']}, which is missing")
    kept = [e for e in entries(session) if e.get("status") == "keep"]
    if kept and (best["run"] != kept[-1]["run_id"]):
        problems.append("best.json does not name the last kept run")
    if session.results_path.read_text(encoding="utf-8") != _results_text(session):
        problems.append("results.tsv does not match the entries (rewrite it with `uv run lab.py export`)")
    return problems
