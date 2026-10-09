import pytest

import capture
import prepare


def test_time_budget_defaults_to_five_minutes():
    assert prepare._time_budget_from_env({}) == 300
    assert prepare._time_budget_from_env({"AUTORESEARCH_TIME_BUDGET": ""}) == 300


def test_a_study_can_set_a_longer_budget():
    assert prepare._time_budget_from_env({"AUTORESEARCH_TIME_BUDGET": "600"}) == 600


@pytest.mark.parametrize("bad", ["ten", "600.5", "30", "4000", "-600"])
def test_a_bad_budget_is_refused_by_name(bad):
    with pytest.raises(ValueError, match="AUTORESEARCH_TIME_BUDGET"):
        prepare._time_budget_from_env({"AUTORESEARCH_TIME_BUDGET": bad})


def test_longer_runs_also_sample_at_five_minutes():
    assert capture.snapshot_times_for(300) == (0, 10, 30, 60, 120)
    assert capture.snapshot_times_for(600) == (0, 10, 30, 60, 120, 300)
