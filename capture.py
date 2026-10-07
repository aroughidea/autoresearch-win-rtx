"""
Run capture for the Training Decisions explorer.

Records what the model writes at fixed moments of training, on fixed prompts with
fixed decoding, and saves one small JSON run file per experiment in runs/.

A captured run trains exactly like an uncaptured one: train.py calls this module
before each step's timer starts (so sampling never counts toward the 5-minute
budget), and sampling uses its own random generator, never the global one.

The agent must not modify this file (see program.md).
"""

import contextlib
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch


def compact_timestamp(moment):
    """2026-05-23 15:57:43 -07:00 -> '20260523T155743-0700'. Colon-free, safe in Windows filenames."""
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.strftime("%Y%m%dT%H%M%S%z")


def make_run_id(commit, moment):
    return f"{compact_timestamp(moment)}_{commit or 'nogit'}"


def git_commit_info(cwd=None):
    """Short hash and commit time of HEAD, or (None, None) outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h%n%cI"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or len(lines) != 2:
        return None, None
    try:
        return lines[0], datetime.fromisoformat(lines[1])
    except ValueError:
        return lines[0], None
