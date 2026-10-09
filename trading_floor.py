"""Trading Floor: a 7-role trading desk sharing one group chat.

Roles: Desk Lead, Market Analyst, Research Analyst, Strategist, Risk Manager,
Execution Trader, Trade Reviewer. PAPER mode only until the human says otherwise.
No exchange is touched here; the Execution Trader works through an injected
broker object and only on human-approved ticket IDs.
Past performance does not guarantee future results.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

MAX_RISK_PER_TRADE = 0.03
MAX_TOTAL_EXPOSURE = 0.10
DAILY_LOSS_LIMIT = 0.05
FUNDING_ANOMALY = 0.30  # annualized
BACKTEST_DAYS = 90
LOG_DEADLINE = timedelta(minutes=10)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GroupChat:
    """The single shared thread. Every agent message goes through here."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def post(self, sender: str, text: str, **data) -> None:
        self.messages.append({"ts": _now(), "from": sender, "text": text, "data": data})


@dataclass
class Ticket:
    id: str
    asset: str
    direction: str  # "long" | "short"
    size: float  # fraction of equity as notional
    entry: float
    invalidation: float
    chain: list[str]
    risk: float = 0.0  # fraction of equity at risk
    status: str = "pending"  # pending | approved | rejected | executed | closed

    def render(self) -> str:
        return (f"[{self.id}] {self.asset} {self.direction} size={self.size:.2%} entry={self.entry} "
                f"invalidation={self.invalidation} chain={' > '.join(self.chain)}")


class Agent:
    role = "Agent"

    def __init__(self, chat: GroupChat) -> None:
        self.chat = chat

    def say(self, text: str, **data) -> None:
        self.chat.post(self.role, text, **data)


class MarketAnalyst(Agent):
    """Data only: never proposes trades."""
    role = "Market Analyst"

    def scan(self, asset: str, funding_annualized: float = 0.0, depth_ratio: float = 1.0,
             oi_change: float = 0.0) -> list[dict]:
        flags = []
        if abs(funding_annualized) > FUNDING_ANOMALY:
            flags.append({"asset": asset, "kind": "funding", "value": funding_annualized})
        if depth_ratio < 0.5:
            flags.append({"asset": asset, "kind": "depth_thinning", "value": depth_ratio})
        if abs(oi_change) > 0.2:
            flags.append({"asset": asset, "kind": "open_interest", "value": oi_change})
        for f in flags:
            self.say(f"ANOMALY {f['asset']} {f['kind']}={f['value']}", **f)
        return flags


class ResearchAnalyst(Agent):
    role = "Research Analyst"

    def __init__(self, chat: GroupChat) -> None:
        super().__init__(chat)
        self.catalysts: list[tuple[datetime, str]] = []

    def add_catalyst(self, when: datetime, label: str) -> None:
        self.catalysts.append((when, label))
        self.say(f"CATALYST {label} at {when.isoformat()}")

    def verify(self, flag: dict, second_source_confirms: bool) -> bool:
        """A flag only moves forward if an independent second source confirms it."""
        self.say(f"{'CONFIRMED' if second_source_confirms else 'UNCONFIRMED'} {flag['asset']} {flag['kind']}")
        return second_source_confirms


class Strategist(Agent):
    role = "Strategist"

    def build(self, flag: dict, direction: str, entry: float, invalidation: float, size: float,
              backtest_ok: bool, out_of_sample_ok: bool) -> dict | None:
        """Both the 90d backtest and an unseen out-of-sample check must pass; else the idea dies here."""
        if not (backtest_ok and out_of_sample_ok):
            self.say(f"KILLED {flag['asset']}: backtest={backtest_ok} out_of_sample={out_of_sample_ok}")
            return None
        idea = {"asset": flag["asset"], "direction": direction, "entry": entry,
                "invalidation": invalidation, "size": size}
        self.say(f"IDEA {flag['asset']} {direction} entry={entry} invalidation={invalidation}", **idea)
        return idea


class RiskManager(Agent):
    """Holds the limits. Hard stops, veto over everyone."""
    role = "Risk Manager"

    def __init__(self, chat: GroupChat) -> None:
        super().__init__(chat)
        self.open_exposure = 0.0
        self.day_pnl = 0.0  # fraction of equity, realised today
        self.paused = False

    def _risk(self, idea: dict) -> float:
        return idea["size"] * abs(idea["entry"] - idea["invalidation"]) / idea["entry"]

    def size(self, idea: dict) -> dict | None:
        if self.paused:
            self.say(f"VETO {idea['asset']}: desk paused for the day")
            return None
        size = idea["size"]
        risk_per_unit = abs(idea["entry"] - idea["invalidation"]) / idea["entry"]
        if risk_per_unit <= 0:
            self.say(f"VETO {idea['asset']}: invalid invalidation level")
            return None
        size = min(size, MAX_RISK_PER_TRADE / risk_per_unit, MAX_TOTAL_EXPOSURE - self.open_exposure)
        if size <= 0:
            self.say(f"VETO {idea['asset']}: no exposure headroom")
            return None
        sized = dict(idea, size=size)
        self.say(f"APPROVED SIZE {idea['asset']} {size:.2%} risk={self._risk(sized):.2%}")
        return sized

    def register_open(self, ticket: Ticket) -> None:
        self.open_exposure += ticket.size

    def register_close(self, ticket: Ticket, pnl: float) -> bool:
        """pnl as fraction of equity. Returns True if the daily loss limit tripped (desk goes flat)."""
        self.open_exposure = max(0.0, self.open_exposure - ticket.size)
        self.day_pnl += pnl
        if self.day_pnl <= -DAILY_LOSS_LIMIT and not self.paused:
            self.paused = True
            self.say("DAILY LOSS LIMIT HIT: desk flat, new tickets paused until next day")
        return self.paused

    def new_day(self) -> None:
        self.day_pnl = 0.0
        self.paused = False


class ExecutionTrader(Agent):
    """The only agent allowed near an exchange connection."""
    role = "Execution Trader"

    def __init__(self, chat: GroupChat, broker=None, mode: str = "paper") -> None:
        super().__init__(chat)
        self._broker = broker
        self.mode = mode

    def execute(self, ticket: Ticket, approved_ids: set[str]) -> bool:
        if ticket.id not in approved_ids or ticket.status != "approved":
            self.say(f"REFUSED {ticket.id}: no human approval for this ticket ID")
            return False
        if self.mode != "paper":
            self.say(f"REFUSED {ticket.id}: only PAPER mode is enabled")
            return False
        perms = set(getattr(self._broker, "permissions", ()) or ())
        if any("withdraw" in p.lower() for p in perms):
            self.say(f"REFUSED {ticket.id}: withdrawal permission detected on API key")
            return False
        if self._broker is not None:
            self._broker.submit(ticket)
        ticket.status = "executed"
        self.say(f"EXECUTED (paper) {ticket.id}")
        return True


@dataclass
class ReviewLog:
    entries: list[dict] = field(default_factory=list)


class TradeReviewer(Agent):
    role = "Trade Reviewer"

    def __init__(self, chat: GroupChat) -> None:
        super().__init__(chat)
        self.log = ReviewLog()

    def log_execution(self, ticket: Ticket, executed_at: datetime, now: datetime | None = None) -> dict:
        now = now or _now()
        entry = {"ticket": ticket.id, "asset": ticket.asset, "direction": ticket.direction,
                 "size": ticket.size, "executed_at": executed_at, "logged_at": now,
                 "late": now - executed_at > LOG_DEADLINE, "post_mortem": None, "pnl": None}
        self.log.entries.append(entry)
        self.say(f"LOGGED {ticket.id}" + (" (LATE)" if entry["late"] else ""))
        return entry

    def post_mortem(self, ticket_id: str, predicted: str, happened: str, gap: str, pnl: float) -> None:
        for e in self.log.entries:
            if e["ticket"] == ticket_id:
                e.update(post_mortem={"predicted": predicted, "happened": happened, "gap": gap}, pnl=pnl)
                self.say(f"POST-MORTEM {ticket_id}: predicted={predicted}; happened={happened}; gap={gap}")
                return
        raise KeyError(ticket_id)

    def weekly_report(self, now: datetime | None = None) -> str:
        """Includes every logged trade, losers too. Entries are never deleted."""
        closed = [e for e in self.log.entries if e["pnl"] is not None]
        wins = sum(1 for e in closed if e["pnl"] > 0)
        text = (f"WEEKLY REPORT trades={len(self.log.entries)} closed={len(closed)} wins={wins} "
                f"losses={len(closed) - wins} pnl={sum(e['pnl'] for e in closed):.4f}")
        self.say(text)
        return text

    @staticmethod
    def is_report_time(now: datetime) -> bool:
        return now.weekday() == 0 and now.hour == 9 and now.minute == 0


class DeskLead(Agent):
    role = "Desk Lead"

    def __init__(self, chat: GroupChat, risk: RiskManager) -> None:
        super().__init__(chat)
        self.risk = risk

    def should_wake_human(self, position_price: float, ticket: Ticket) -> bool:
        """Overnight: only wake the human when an open position's invalidation is hit."""
        if ticket.status != "executed":
            return False
        hit = (position_price <= ticket.invalidation if ticket.direction == "long"
               else position_price >= ticket.invalidation)
        if hit:
            self.say(f"INVALIDATION HIT {ticket.id}: waking human")
        return hit


class TradingFloor:
    """Wires the seven agents together around one shared chat."""

    def __init__(self, broker=None, mode: str = "paper") -> None:
        self.chat = GroupChat()
        self.market = MarketAnalyst(self.chat)
        self.research = ResearchAnalyst(self.chat)
        self.strategist = Strategist(self.chat)
        self.risk = RiskManager(self.chat)
        self.trader = ExecutionTrader(self.chat, broker, mode)
        self.reviewer = TradeReviewer(self.chat)
        self.lead = DeskLead(self.chat, self.risk)
        self.tickets: dict[str, Ticket] = {}
        self.approved: set[str] = set()
        self._ids = itertools.count(1)

    def propose(self, flag: dict, second_source_confirms: bool, direction: str, entry: float,
                invalidation: float, size: float, backtest_ok: bool, out_of_sample_ok: bool) -> Ticket | None:
        chain = [self.market.role]
        if not self.research.verify(flag, second_source_confirms):
            return None
        chain.append(self.research.role)
        idea = self.strategist.build(flag, direction, entry, invalidation, size, backtest_ok, out_of_sample_ok)
        if idea is None:
            return None
        chain.append(self.strategist.role)
        sized = self.risk.size(idea)
        if sized is None:
            return None
        chain.append(self.risk.role)
        t = Ticket(f"T{next(self._ids):04d}", sized["asset"], direction, sized["size"], entry,
                   invalidation, chain, risk=self.risk._risk(sized))
        self.tickets[t.id] = t
        self.lead.say(f"TICKET FOR HUMAN {t.render()}")
        return t

    def human_decision(self, ticket_id: str, approve: bool) -> None:
        """Approval is by ticket ID only."""
        t = self.tickets[ticket_id]
        if t.status != "pending":
            return
        t.status = "approved" if approve else "rejected"
        if approve:
            self.approved.add(ticket_id)
        self.lead.say(f"HUMAN {'APPROVED' if approve else 'REJECTED'} {ticket_id}")

    def execute(self, ticket_id: str) -> bool:
        t = self.tickets[ticket_id]
        if not self.trader.execute(t, self.approved):
            return False
        self.risk.register_open(t)
        self.reviewer.log_execution(t, _now())
        return True

    def close(self, ticket_id: str, pnl: float) -> None:
        t = self.tickets[ticket_id]
        t.status = "closed"
        self.risk.register_close(t, pnl)

    def propose_from_desk(self, decision: dict, entry: float, invalidation: float,
                          second_source_confirms: bool, size: float = 0.10) -> Ticket | None:
        """Feed a MultiAgentDesk.analyze() result through the floor.

        The desk's selection Sharpe is treated as the backtest check and its
        holdout Sharpe as the out-of-sample check. FLAT decisions are ignored.
        """
        if decision.get("action") not in ("LONG", "SHORT"):
            return None
        flag = {"asset": decision["symbol"], "kind": "desk_signal", "value": decision.get("consensus")}
        return self.propose(
            flag, second_source_confirms, decision["action"].lower(), entry, invalidation, size,
            backtest_ok=decision.get("select_sharpe", 0) > 0,
            out_of_sample_ok=decision.get("holdout_sharpe", 0) > 0)
