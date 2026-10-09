import pytest

from trading_floor import TradingFloor


def _propose(f, **kw):
    args = dict(flag={"asset": "BTC", "kind": "funding", "value": 0.4}, second_source_confirms=True,
                direction="long", entry=100.0, invalidation=98.0, size=0.5,
                backtest_ok=True, out_of_sample_ok=True)
    args.update(kw)
    return f.propose(**args)


def test_size_capped_by_risk_and_exposure():
    f = TradingFloor()
    t = _propose(f)
    assert t.size == pytest.approx(0.10)  # exposure cap binds (risk cap would give 1.5)
    assert t.risk <= 0.03 + 1e-9
    t2 = _propose(f, invalidation=50.0, size=0.5)
    assert t2.size == pytest.approx(0.06)  # 3% risk / 50% stop distance


def test_strategist_kills_failed_ideas():
    f = TradingFloor()
    assert _propose(f, out_of_sample_ok=False) is None
    assert _propose(f, second_source_confirms=False) is None


def test_no_execution_without_id_approval():
    f = TradingFloor()
    t = _propose(f)
    assert not f.execute(t.id)
    f.human_decision(t.id, True)
    assert f.execute(t.id)
    assert f.reviewer.log.entries[0]["ticket"] == t.id


def test_withdraw_permission_and_live_mode_refused():
    class B:
        permissions = ["trade", "withdraw"]

        def submit(self, t):
            raise AssertionError

    f = TradingFloor(broker=B())
    t = _propose(f)
    f.human_decision(t.id, True)
    assert not f.execute(t.id)
    g = TradingFloor(mode="live")
    t = _propose(g)
    g.human_decision(t.id, True)
    assert not g.execute(t.id)


def test_daily_loss_pauses_desk_and_wakes_human_only_on_invalidation():
    f = TradingFloor()
    t = _propose(f)
    f.human_decision(t.id, True)
    f.execute(t.id)
    assert not f.lead.should_wake_human(99.0, t)
    assert f.lead.should_wake_human(98.0, t)
    f.close(t.id, -0.05)
    assert _propose(f) is None


def test_chat_is_shared_and_ticket_has_chain():
    f = TradingFloor()
    t = _propose(f)
    assert t.chain[0] == "Market Analyst" and "Risk Manager" in t.chain
    assert {m["from"] for m in f.chat.messages} >= {"Research Analyst", "Strategist", "Risk Manager"}


def test_propose_from_desk():
    f = TradingFloor()
    d = {"symbol": "SPY", "action": "LONG", "consensus": 0.6, "select_sharpe": 1.1, "holdout_sharpe": 0.4}
    t = f.propose_from_desk(d, entry=100.0, invalidation=98.0, second_source_confirms=True)
    assert t.asset == "SPY" and t.direction == "long"
    assert f.propose_from_desk(dict(d, holdout_sharpe=-0.1), 100.0, 98.0, True) is None
    assert f.propose_from_desk(dict(d, action="FLAT"), 100.0, 98.0, True) is None
