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

    def __init__(self, clock, minutes=60, result=None, init=None):
        self.clock, self.minutes, self.calls, self.init = clock, minutes, [], init
        self.result = result or {"type": "result", "is_error": False, "result": "done"}

    def __call__(self, cmd, prompt, env, stdout, stderr):
        self.calls.append((cmd, prompt))
        if self.init:
            stdout.write(json.dumps(self.init) + "\n")
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


def test_a_continued_session_starts_with_the_continue_prompt(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch, hours=3)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=60)
    _run(tmp_path, s, agent, clock, continuing=True)
    first, again = session.prompts(s)
    resumed = session.continue_prompt(s)
    assert agent.calls[0][1] == resumed and "--continue" not in agent.calls[0][0]
    assert agent.calls[1][1] == again and "--continue" in agent.calls[1][0]
    assert "do not start a new one" in resumed and "lab.py history" in resumed and "did not finish" in resumed
    assert "start a new experiment loop" not in resumed and s.name in resumed


def test_a_given_model_is_passed_to_every_launch(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch, hours=2)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=60)
    _run(tmp_path, s, agent, clock, model="claude-opus-5-5")
    assert all(c[c.index("--model") + 1] == "claude-opus-5-5" for c, _ in agent.calls)


def test_the_model_the_agent_ran_with_is_recorded(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch, hours=1)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=30, init={"type": "system", "subtype": "init", "model": "claude-fable-5"})
    _run(tmp_path, s, agent, clock)
    meta = record.Session(s.root).meta
    assert meta["agent_model"] == "claude-fable-5" and len(meta["agent_models"]) == 1


def test_ctrl_c_ends_the_agent_and_says_so(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=600)
    killed = []

    def sleep(seconds):
        raise KeyboardInterrupt

    reason = session.run_agent(s, logs=tmp_path / "logs", launch=agent, clock=clock.time, sleep=sleep,
                               kill=killed.append)
    assert killed and "Ctrl+C" in reason


def test_a_stop_request_ends_the_agent(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=600)
    killed = []
    stop = tmp_path / "stop"

    def sleep(seconds):
        clock.sleep(seconds)
        stop.write_text("", encoding="utf-8")

    reason = session.run_agent(s, logs=tmp_path / "logs", launch=agent, clock=clock.time, sleep=sleep,
                               kill=killed.append, stop_file=stop)
    assert killed and "--stop" in reason and len(agent.calls) == 1


def test_stop_with_no_runner_says_so(tmp_path):
    message = session.request_stop(tmp_path / "runner.lock", alive=lambda pid: False, sleep=lambda s: None)
    assert "No session runner" in message


def test_stop_asks_a_live_runner_and_waits_for_it(tmp_path):
    lock = tmp_path / "runner.lock"
    lock.mkdir()
    (lock / "runner.pid").write_text("111", encoding="utf-8")

    def sleep(seconds):                       # the runner sees the stop file and finishes
        assert (lock / "stop").exists()
        for f in lock.iterdir():
            f.unlink()
        lock.rmdir()

    message = session.request_stop(lock, alive=lambda pid: pid == 111, sleep=sleep)
    assert "Stopped" in message and not lock.exists()


def test_stop_clears_a_dead_runners_lock_and_names_its_agent(tmp_path):
    lock = tmp_path / "runner.lock"
    lock.mkdir()
    (lock / "runner.pid").write_text("111", encoding="utf-8")
    (lock / "agent.pid").write_text("222", encoding="utf-8")
    message = session.request_stop(lock, alive=lambda pid: pid == 222, sleep=lambda s: None)
    assert not lock.exists() and "222" in message and "no longer running" in message


def test_check_names_the_agents_model_or_says_the_default_is_recorded():
    kw = dict(run_minutes=10, hours=10, env={}, which={"claude": "c", "uv": "u"}.get, signin=lambda: "ok",
              disk_free_gb=lambda: 100, locked=lambda: False)
    pinned = [m for _, m in session.checks(model="claude-opus-5-5", **kw)]
    default = [m for _, m in session.checks(**kw)]
    assert any("claude-opus-5-5" in m for m in pinned)
    assert any("default model" in m and "record" in m for m in default)


def test_an_unexpected_error_still_ends_the_agent(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch)
    clock = _Clock(STARTED)
    killed = []

    class _Breaks:
        pid = 4242

        def poll(self):
            raise RuntimeError("something unexpected")

    with pytest.raises(RuntimeError):
        session.run_agent(s, logs=tmp_path / "logs", launch=lambda *a: _Breaks(), clock=clock.time,
                          sleep=clock.sleep, kill=killed.append)
    assert killed


def test_a_locked_session_json_or_a_missing_lock_folder_does_not_stop_the_runner(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch, hours=1)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=30, init={"type": "system", "subtype": "init", "model": "claude-fable-5"})

    def locked(*a, **kw):
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(record, "note_agent_model", locked)
    reason = session.run_agent(s, logs=tmp_path / "logs", launch=agent, clock=clock.time, sleep=clock.sleep,
                               kill=lambda proc: None, stop_file=tmp_path / "gone" / "stop")
    assert "time is up" in reason


def test_a_stop_request_cuts_a_long_pause_short(tmp_path, monkeypatch):
    s = _project(tmp_path, monkeypatch)
    clock = _Clock(STARTED)
    agent = _Agent(clock, minutes=1, result={"type": "result", "is_error": True, "result": "usage limit reached"})
    stop = tmp_path / "stop"
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock.sleep(seconds)
        if sum(slept) >= 60:
            stop.write_text("", encoding="utf-8")

    reason = session.run_agent(s, logs=tmp_path / "logs", launch=agent, clock=clock.time, sleep=sleep,
                               kill=lambda proc: None, stop_file=stop)
    assert "--stop" in reason and sum(slept) < 900


def test_stop_survives_the_runner_finishing_at_that_moment(tmp_path):
    lock = tmp_path / "runner.lock"
    lock.mkdir()
    (lock / "runner.pid").write_text("111", encoding="utf-8")

    def alive(pid):                           # the runner removes its lock just as we look
        for f in lock.iterdir():
            f.unlink()
        lock.rmdir()
        return True

    assert "Stopped" in session.request_stop(lock, alive=alive, sleep=lambda s: None)


def test_on_posix_the_agent_gets_its_own_process_group_and_the_group_is_ended(monkeypatch):
    seen = {}

    class _Popen:
        def __init__(self, cmd, **kw):
            seen.update(kw)
            self.pid = 777
            self.stdin = type("S", (), {"write": lambda self, t: None, "close": lambda self: None})()

    monkeypatch.setattr(session.os, "name", "posix")
    monkeypatch.setattr(session.subprocess, "Popen", _Popen)
    proc = session._launch(["claude"], "prompt", {}, None, None)
    assert seen.get("start_new_session") is True
    groups = []
    monkeypatch.setattr(session.os, "getpgid", lambda pid: pid + 1, raising=False)
    monkeypatch.setattr(session.os, "killpg", lambda pgid, sig: groups.append((pgid, sig)), raising=False)
    session._kill(proc)
    assert groups and groups[0][0] == 778
