import ctypes
import sys

import pytest

import prepare

PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
EXECUTION_SPEED = 0x1         # PROCESS_POWER_THROTTLING_EXECUTION_SPEED


class FakeKernel32:
    """Stands in for kernel32: records each SetProcessInformation call and gives a set answer."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def GetCurrentProcess(self):
        return 7

    def SetProcessInformation(self, handle, info_class, state_ref, size):
        state = state_ref._obj
        self.calls.append((handle, info_class, state.Version, state.ControlMask, state.StateMask, size))
        return self.answer


def test_it_asks_windows_to_stop_throttling(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    fake = FakeKernel32(answer=1)
    assert prepare.opt_out_of_power_throttling(fake) is True
    # Version 1, control the execution-speed policy, and turn it off; the state is three ULONGs.
    assert fake.calls == [(7, PROCESS_POWER_THROTTLING, 1, EXECUTION_SPEED, 0, 12)]


def test_a_refusal_returns_false_and_does_not_raise(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    assert prepare.opt_out_of_power_throttling(FakeKernel32(answer=0)) is False
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


def test_off_windows_nothing_is_called(monkeypatch, capsys):
    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"kernel32.{name} called off Windows")

    monkeypatch.setattr(sys, "platform", "linux")
    assert prepare.opt_out_of_power_throttling(Untouchable()) is False
    assert capsys.readouterr().out == ""


class _ThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


@pytest.mark.skipif(sys.platform != "win32", reason="power throttling is a Windows policy")
def test_importing_prepare_opts_this_process_out():
    # prepare was imported above, as train.py imports it: Windows should now report the
    # execution-speed policy as controlled by this process and switched off.
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.GetProcessInformation.restype = wintypes.BOOL
    state = _ThrottlingState(1, 0, 0)
    ok = kernel32.GetProcessInformation(kernel32.GetCurrentProcess(), PROCESS_POWER_THROTTLING,
                                        ctypes.byref(state), ctypes.sizeof(state))
    assert ok, f"GetProcessInformation failed (error {ctypes.get_last_error()})"
    assert state.ControlMask & EXECUTION_SPEED
    assert not state.StateMask & EXECUTION_SPEED
