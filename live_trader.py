"""Autonomous trader: the head agent's decisions -> Alpaca orders, running 24/7.

SAFETY DEFAULTS
  * Paper trading unless BOTH --live is passed AND env LIVE_TRADING=I_UNDERSTAND is set.
  * Long-only by default (--allow-short to enable; Alpaca does not short crypto).
  * Kill switch: create a file named KILL next to this script -> flatten all and exit.
  * Daily loss halt: if equity falls more than --max-daily-loss vs prior close, flatten and exit.
  * Per-symbol cap, minimum trade size, and halt after repeated consecutive errors.

Alpaca trades US stocks/ETFs and crypto only. Forex and futures from the desk universe are skipped.
This is not financial advice. Past performance does not guarantee future results. You can lose money.
Run in paper mode for weeks before risking real capital.

Env: ALPACA_KEY_ID, ALPACA_SECRET_KEY
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from multi_agent import MultiAgentDesk, load_ohlc

log = logging.getLogger("live_trader")

PAPER_URL = "https://paper-api.alpaca.markets/v2"
LIVE_URL = "https://api.alpaca.markets/v2"
KILL_FILE = Path(__file__).with_name("KILL")
TRADABLE = {"stock", "crypto"}


# ----------------------------------------------------------------------------
# Pure sizing logic (unit-tested)
# ----------------------------------------------------------------------------
def target_notional(position: float, equity: float, n_symbols: int, max_symbol_frac: float,
                    allow_short: bool) -> float:
    """Desk position in [-1,1] -> dollar target for one symbol."""
    if not allow_short:
        position = max(position, 0.0)
    position = max(-1.0, min(1.0, position))
    cap = min(1.0 / max(n_symbols, 1), max_symbol_frac)
    return position * cap * equity


def order_for(target: float, current: float, min_trade: float) -> tuple[str, float] | None:
    """Return (side, notional) or None if the change is too small to bother with."""
    diff = target - current
    if abs(diff) < min_trade:
        return None
    return ("buy" if diff > 0 else "sell", round(abs(diff), 2))


def to_alpaca(symbol: str, asset_class: str) -> str:
    return symbol.replace("-USD", "/USD") if asset_class == "crypto" else symbol


def pos_key(symbol: str) -> str:
    return symbol.replace("/", "")


# ----------------------------------------------------------------------------
# Broker
# ----------------------------------------------------------------------------
class Alpaca:
    def __init__(self, live: bool):
        self.base = LIVE_URL if live else PAPER_URL
        self.s = requests.Session()
        self.s.headers.update({
            "APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
        })

    def _req(self, method: str, path: str, **kw):
        r = self.s.request(method, self.base + path, timeout=15, **kw)
        r.raise_for_status()
        return r.json() if r.content else None

    def account(self):
        return self._req("GET", "/account")

    def clock(self):
        return self._req("GET", "/clock")

    def positions(self) -> dict[str, float]:
        return {p["symbol"]: float(p["market_value"]) for p in self._req("GET", "/positions")}

    def order(self, symbol: str, side: str, notional: float, crypto: bool):
        body = {"symbol": symbol, "side": side, "type": "market",
                "notional": str(notional), "time_in_force": "gtc" if crypto else "day"}
        return self._req("POST", "/orders", json=body)

    def flatten(self):
        return self._req("DELETE", "/positions", params={"cancel_orders": "true"})


# ----------------------------------------------------------------------------
# Trader
# ----------------------------------------------------------------------------
class LiveTrader:
    def __init__(self, broker: Alpaca, universe: dict[str, list[str]], args):
        self.b = broker
        self.universe = {k: v for k, v in universe.items() if k in TRADABLE}
        self.args = args
        self.desk = MultiAgentDesk()
        self.decisions: dict[str, tuple[str, float]] = {}  # alpaca symbol -> (asset_class, position)
        self.decided_on: str | None = None
        self.errors = 0

    def refresh_decisions(self):
        """Desk uses daily bars, so recompute once per UTC day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.decided_on == today:
            return
        new = {}
        for ac, syms in self.universe.items():
            for sym in syms:
                try:
                    r = self.desk.analyze(sym, ac, load_ohlc(sym))
                    new[to_alpaca(sym, ac)] = (ac, float(r["position"]))
                    log.info("decision %s: %s via %s (agree %s/7)", sym, r["action"],
                             r["best_strategy"], r["agents_agree"])
                except Exception as exc:
                    log.warning("analysis failed for %s: %s", sym, exc)
        if new:
            self.decisions, self.decided_on = new, today

    def halt(self, reason: str):
        log.critical("HALT: %s -- flattening all positions", reason)
        try:
            self.b.flatten()
        except Exception:
            log.exception("flatten FAILED -- close positions manually NOW")
        sys.exit(1)

    def risk_checks(self, acct: dict):
        if KILL_FILE.exists():
            self.halt("kill switch file present")
        equity, last = float(acct["equity"]), float(acct["last_equity"])
        if last > 0 and equity / last - 1.0 < -self.args.max_daily_loss:
            self.halt(f"daily loss {equity / last - 1:.2%} breached limit {-self.args.max_daily_loss:.2%}")
        if acct.get("trading_blocked") or acct.get("account_blocked"):
            self.halt("account blocked by broker")

    def cycle(self):
        acct = self.b.account()
        self.risk_checks(acct)
        self.refresh_decisions()
        if not self.decisions:
            log.warning("no decisions yet; skipping cycle")
            return
        equity = float(acct["equity"])
        held = self.b.positions()
        stocks_open = self.b.clock()["is_open"]
        n = len(self.decisions)
        for sym, (ac, pos) in self.decisions.items():
            if ac == "stock" and not stocks_open:
                continue
            tgt = target_notional(pos, equity, n, self.args.max_symbol_frac, self.args.allow_short)
            cur = held.get(pos_key(sym), 0.0)
            o = order_for(tgt, cur, self.args.min_trade)
            if o is None:
                continue
            side, notional = o
            log.info("ORDER %s %s $%.2f (target %.2f, current %.2f)", side, sym, notional, tgt, cur)
            self.b.order(sym, side, notional, crypto=(ac == "crypto"))

    def run(self):
        log.info("started; universe=%s", self.universe)
        while True:
            try:
                self.cycle()
                self.errors = 0
            except SystemExit:
                raise
            except Exception:
                self.errors += 1
                log.exception("cycle error %d/%d", self.errors, self.args.max_errors)
                if self.errors >= self.args.max_errors:
                    self.halt("too many consecutive errors")
            time.sleep(self.args.interval)


def main():
    ap = argparse.ArgumentParser(description="24/7 autonomous trading (paper by default)")
    ap.add_argument("--live", action="store_true", help="real money; also requires LIVE_TRADING=I_UNDERSTAND")
    ap.add_argument("--interval", type=int, default=900, help="seconds between cycles")
    ap.add_argument("--max-daily-loss", type=float, default=0.03)
    ap.add_argument("--max-symbol-frac", type=float, default=0.25)
    ap.add_argument("--min-trade", type=float, default=25.0, help="min $ change to trade")
    ap.add_argument("--max-errors", type=int, default=5)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--stocks", default="AAPL,MSFT,SPY")
    ap.add_argument("--crypto", default="BTC-USD,ETH-USD")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("live_trader.log")])
    if args.live and os.environ.get("LIVE_TRADING") != "I_UNDERSTAND":
        sys.exit("Refusing live mode: set LIVE_TRADING=I_UNDERSTAND to confirm.")
    for k in ("ALPACA_KEY_ID", "ALPACA_SECRET_KEY"):
        if k not in os.environ:
            sys.exit(f"Missing env var {k}")

    log.warning("MODE: %s", "LIVE (real money)" if args.live else "PAPER")
    uni = {"stock": [s for s in args.stocks.split(",") if s], "crypto": [s for s in args.crypto.split(",") if s]}
    LiveTrader(Alpaca(args.live), uni, args).run()


if __name__ == "__main__":
    main()
