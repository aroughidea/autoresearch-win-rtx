"""session.py: the runner for an unattended session, with a fake agent, a fake clock and no GPU."""
import datetime as dt
import json

import pytest

import record
import session

STARTED = dt.datetime(2026, 10, 9, 20, 0, 0).astimezone()


def _project(tmp_path, monkeypatch, hours=10, run_minutes=10):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.py").write_text("MATRIX_LR = 0.05\n", encoding="utf-8")
    for name in record.FIXED_FILES:
        (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
    return record.start_session("tinystories-own-10min-2026-10-09", dataset="tinystories", tokenizer="own",
                                run_minutes=run_minutes, hours=hours, now=STARTED)


class _Clock:
    def __init__(self, start):
        self.now = start.timestamp()

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _Agent:
    """Each launch 'runs' for a set time on the fake clock, then writes a result line like claude's."""

    def __init__(self, clock, minutes=60, result=None):
        self.clock, self.minutes, self.calls = clock, minutes, []
        self.result = result or {"type": "result", "is_error": False, "result": "done"}

    def __call__(self, cmd, prompt, env, stdout, stderr):
        self.calls.append((cmd, prompt))
        stdout.write(json.dumps(self.result) + "\n")
        stdout.flush()
        clock, until = self.clock, self.clock.now + self.minutes * 60

        class _Proc:
            pid = 4242

            def poll(self):
                return 0 if clock.now >= until else None

            def wait(self):
                return 0

        return _Proc()


def _run(tmp_path, s, agent, clock, **kw):
    return session.run_agent(s, logs=tmp_path / "session-logs" / s.name, launch=agent,
                             clock=clock.time, sleep=clock.sleep, kill=lambda proc: None, **kw)


def test_prompts_name_the_session_its_end_and_the_record(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch)
    first, again = session.prompts(s)
    assert s.name in first and "lab.py keep" in first and "never use git" in first
    assert "06:00" in first and "headless" in first
    assert "lab.py status" in again and "06:00" in again
    assert "AUTORESEARCH_TIME_BUDGET" not in first          # 10-minute runs are the default


def test_a_session_of_other_run_lengths_says_so_and_sets_the_budget(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch, run_minutes=5)
    first, _ = session.prompts(s)
    assert "5 minutes" in first and "AUTORESEARCH_TIME_BUDGET=300" in first
    assert session.agent_env(s, {"PATH": "x"})["AUTORESEARCH_TIME_BUDGET"] == "300"
    s10 = record.Session(s.root)
    s10.meta["run_minutes"] = 10
    assert "AUTORESEARCH_TIME_BUDGET" not in session.agent_env(s10, {"AUTORESEARCH_TIME_BUDGET": "300"})


def test_the_agent_is_relaunched_with_continue_until_the_last_20_minutes(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch, hours=3)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=60)
    reason = _run(tmp_path, s, agent, clock)
    commands = [c for c, _ in agent.calls]
    first, again = session.prompts(s)
    assert commands[0][:2] == ["claude", "--dangerously-skip-permissions"] and "--continue" not in commands[0]
    assert all("--continue" in c for c in commands[1:])
    assert agent.calls[0][1] == first and agent.calls[1][1] == again     # the prompt goes in on stdin
    assert len(commands) == 3 and "time is up" in reason             # 3 h, last launch by 2 h 40
    assert clock.now <= (STARTED + dt.timedelta(hours=3, minutes=25)).timestamp()


def test_a_sign_in_failure_stops_at_once(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=1, result={"type": "result", "is_error": True,
                                             "result": "Invalid API key · Please run /login (401)"})
    reason = _run(tmp_path, s, agent, clock)
    assert len(agent.calls) == 1 and "sign-in" in reason


def test_three_quick_endings_in_a_row_stop(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=1, result={"type": "result", "is_error": True, "result": "something broke"})
    reason = _run(tmp_path, s, agent, clock)
    assert len(agent.calls) == 3 and "three launches" in reason


def test_the_hard_stop_ends_an_agent_that_runs_on(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch, hours=1)
    clock = _Clock(STARTED)
    killed = []
    agent = _Agent(clock, minutes=600)
    reason = session.run_agent(s, logs=tmp_path / "logs", launch=agent, clock=clock.time, sleep=clock.sleep,
                               kill=killed.append)
    assert killed and "hard stop" in reason
    assert clock.now <= (STARTED + dt.timedelta(hours=1, minutes=26)).timestamp()


def test_check_reports_what_is_missing():
    found = {"uv": "/bin/uv"}
    results = session.checks(run_minutes=10, hours=10, env={"AUTORESEARCH_TIME_BUDGET": "300"},
                             which=found.get, signin=lambda: "ok", disk_free_gb=lambda: 100,
                             locked=lambda: False)
    failed = [message for ok, message in results if not ok]
    assert any("claude" in m for m in failed) and any("AUTORESEARCH_TIME_BUDGET" in m for m in failed)


def test_the_session_name_has_the_pair_the_length_and_the_date():
    assert session.session_name("folktales", "phi3", 5, dt.date(2026, 10, 9)) == "folktales-phi3-5min-2026-10-09"
