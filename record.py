"""
The experiment record: one folder per session (sessions/<name>/) holding every version of
train.py that trained (versions/), one entry per run (runs/), and a pointer to the best version
(best.json). lab.py is its command line; capture.py calls begin_run() as each training run starts.

Standard library only, so lab.py starts fast. The agent must not modify this file (see program.md).
"""

import datetime as _dt
import hashlib
import json
import os
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


def write_results_tsv(session):
    """results.tsv, generated from the entries. Completed in Task 2."""
    write_atomic(session.results_path, "\t".join(RESULTS_HEADER) + "\n")
