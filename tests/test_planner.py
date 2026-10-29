import pytest

from jevbot import planner


def test_first_month_splits_by_target():
    p = planner.monthly({}, 10000, month=1)
    assert p.buys == pytest.approx({"nifty": 3500, "gold": 3500, "crypto": 3000})


def test_new_money_tops_up_what_is_behind_without_selling():
    p = planner.monthly({"nifty": 5000, "gold": 2000, "crypto": 3000}, 3000, month=6)
    assert p.buys["gold"] > p.buys["nifty"] and all(v >= 0 for v in p.buys.values())
    assert sum(p.buys.values()) == pytest.approx(3000)
    assert p.rebalance == {}                                   # never sells outside April


def test_april_proposes_rebalance_only_when_far_off():
    far = planner.monthly({"nifty": 1000, "gold": 1000, "crypto": 20000}, 1000, month=4)
    assert far.rebalance and far.rebalance["crypto"] < 0      # sell some crypto
    assert sum(far.rebalance.values()) == pytest.approx(0, abs=1e-6)
    near = planner.monthly({"nifty": 3500, "gold": 3500, "crypto": 3000}, 1000, month=4)
    assert near.rebalance == {}


def test_goal_maths():
    assert planner.capital_for_daily(10, 0.20) == pytest.approx(18250)
    assert planner.years_to_goal(0, 1000, 0.0, 12000) == pytest.approx(1.0)
    assert planner.years_to_goal(0, 0, 0.1, 100) is None


def test_report_renders():
    text = planner.report({"nifty": 0, "gold": 0, "crypto": 0}, 5000, 4, usd_inr=88)
    assert "Nifty" in text and "Goal" in text and "years" in text
