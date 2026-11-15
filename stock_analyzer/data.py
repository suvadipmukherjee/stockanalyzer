"""Data layer: pull prices and fundamentals from yfinance and boil them down to
a compact, structured snapshot that Claude can reason over.

Claude never sees raw DataFrames. We compute the numbers here (returns,
moving averages, volatility, drawdown, valuation ratios) so the model
interprets facts instead of doing arithmetic on a wall of text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Valuation / quality fields we pull from yfinance's `info` dict.
# key in snapshot -> key in yfinance info
INFO_FIELDS = {
    "company_name": "longName",
    "sector": "sector",
    "industry": "industry",
    "currency": "currency",
    "market_cap": "marketCap",
    "enterprise_value": "enterpriseValue",
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "peg_ratio": "trailingPegRatio",
    "price_to_sales": "priceToSalesTrailing12Months",
    "price_to_book": "priceToBook",
    "ev_to_ebitda": "enterpriseToEbitda",
    "ev_to_revenue": "enterpriseToRevenue",
    "profit_margin": "profitMargins",
    "operating_margin": "operatingMargins",
    "gross_margin": "grossMargins",
    "return_on_equity": "returnOnEquity",
    "revenue_growth_yoy": "revenueGrowth",
    "earnings_growth_yoy": "earningsGrowth",
    "debt_to_equity": "debtToEquity",
    "current_ratio": "currentRatio",
    "free_cash_flow": "freeCashflow",
    "dividend_yield": "dividendYield",
    "beta": "beta",
    "analyst_recommendation": "recommendationKey",
    "analyst_target_mean": "targetMeanPrice",
    "analyst_count": "numberOfAnalystOpinions",
}


@dataclass
class Snapshot:
    ticker: str
    as_of: str
    profile: dict[str, Any] = field(default_factory=dict)
    price: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    valuation: dict[str, Any] = field(default_factory=dict)
    quarterly: list[dict[str, Any]] = field(default_factory=list)
    recent_closes: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- helpers

def _clean(v: Any) -> Any:
    """Make values JSON-friendly and drop NaN/inf."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _pct_return(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return _clean(close.iloc[-1] / close.iloc[-1 - days] - 1)


def price_and_risk_stats(close: pd.Series, benchmark: pd.Series | None = None) -> tuple[dict, dict]:
    """Trend and risk metrics from a daily close series."""
    close = close.dropna()
    last = close.iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    window = close.iloc[-TRADING_DAYS:]

    price = {
        "last_close": _clean(last),
        "last_date": close.index[-1].strftime("%Y-%m-%d"),
        "return_1m": _pct_return(close, 21),
        "return_3m": _pct_return(close, 63),
        "return_6m": _pct_return(close, 126),
        "return_1y": _pct_return(close, 251),
        "sma_50": _clean(sma50),
        "sma_200": _clean(sma200),
        "price_vs_sma50": _clean(last / sma50 - 1) if sma50 else None,
        "price_vs_sma200": _clean(last / sma200 - 1) if sma200 else None,
        "golden_cross": bool(sma50 > sma200) if sma50 and sma200 else None,
        "high_52w": _clean(window.max()),
        "low_52w": _clean(window.min()),
        "pct_below_52w_high": _clean(last / window.max() - 1),
    }

    rets = close.pct_change().dropna()
    running_max = close.cummax()
    drawdown = close / running_max - 1
    downside = rets[rets < 0]
    ann_ret = (1 + rets.mean()) ** TRADING_DAYS - 1
    ann_vol = rets.std() * math.sqrt(TRADING_DAYS)
    risk = {
        "lookback_trading_days": len(close),
        "annualized_volatility": _clean(ann_vol),
        "volatility_last_30d": _clean(rets.iloc[-21:].std() * math.sqrt(TRADING_DAYS)),
        "max_drawdown": _clean(drawdown.min()),
        "current_drawdown": _clean(drawdown.iloc[-1]),
        "sharpe_ratio_rf0": _clean(ann_ret / ann_vol) if ann_vol else None,
        "sortino_ratio_rf0": _clean(ann_ret / (downside.std() * math.sqrt(TRADING_DAYS)))
        if len(downside) > 1 else None,
        "worst_day": _clean(rets.min()),
        "best_day": _clean(rets.max()),
        "value_at_risk_95_daily": _clean(np.percentile(rets, 5)),
    }

    if benchmark is not None and len(benchmark) > 30:
        b = benchmark.dropna().pct_change().dropna()
        joined = pd.concat([rets, b], axis=1, join="inner").dropna()
        if len(joined) > 30:
            cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
            risk["beta_vs_benchmark_computed"] = _clean(cov[0, 1] / cov[1, 1])
            risk["correlation_vs_benchmark"] = _clean(joined.corr().iloc[0, 1])
            bench_close = benchmark.dropna()
            risk["benchmark_return_1y"] = _pct_return(bench_close, 251)

    return price, risk


def _quarterly(t) -> list[dict]:
    """Last 4 quarters of revenue / net income / EPS where available."""
    try:
        q = t.quarterly_income_stmt
    except Exception:
        return []
    if q is None or q.empty:
        return []
    rows = {
        "revenue": ["Total Revenue", "Operating Revenue"],
        "net_income": ["Net Income", "Net Income Common Stockholders"],
        "diluted_eps": ["Diluted EPS", "Basic EPS"],
        "operating_income": ["Operating Income"],
    }
    out = []
    for col in list(q.columns)[:4]:
        entry = {"quarter_end": pd.Timestamp(col).strftime("%Y-%m-%d")}
        for name, candidates in rows.items():
            for c in candidates:
                if c in q.index:
                    entry[name] = _clean(q.loc[c, col])
                    break
        out.append(entry)
    return out


# ---------------------------------------------------------------- public API

def fetch_snapshot(ticker: str, period: str = "2y", benchmark: str = "SPY") -> Snapshot:
    """Fetch live data for `ticker` via yfinance and build a Snapshot."""
    import yfinance as yf  # imported lazily so --sample mode works offline

    ticker = ticker.upper()
    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No price history returned for {ticker!r}. Is the symbol right?")

    bench_close = None
    if benchmark:
        try:
            bench_close = yf.Ticker(benchmark).history(period=period, auto_adjust=True)["Close"]
        except Exception:
            bench_close = None

    try:
        info = t.info or {}
    except Exception:
        info = {}

    return build_snapshot(ticker, hist, info, bench_close, _quarterly(t), benchmark)


def build_snapshot(
    ticker: str,
    hist: pd.DataFrame,
    info: dict,
    bench_close: pd.Series | None = None,
    quarterly: list[dict] | None = None,
    benchmark: str = "SPY",
) -> Snapshot:
    """Pure function (no network) so it can be tested with fixtures."""
    close = hist["Close"]
    price, risk = price_and_risk_stats(close, bench_close)

    snap = Snapshot(ticker=ticker, as_of=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    for key, src in INFO_FIELDS.items():
        val = _clean(info.get(src))
        if key in ("company_name", "sector", "industry", "currency"):
            snap.profile[key] = val
        elif key == "beta":
            risk["beta_reported"] = val
        else:
            snap.valuation[key] = val
    if snap.valuation.get("peg_ratio") is None and info.get("pegRatio") is not None:
        snap.valuation["peg_ratio"] = _clean(info.get("pegRatio"))

    target = snap.valuation.get("analyst_target_mean")
    if target and price.get("last_close"):
        snap.valuation["upside_to_analyst_target"] = _clean(target / price["last_close"] - 1)

    snap.price = price
    snap.risk = risk
    snap.quarterly = quarterly or []
    snap.recent_closes = [
        {"date": d.strftime("%Y-%m-%d"), "close": _clean(c)}
        for d, c in close.iloc[-10:].items()
    ]

    missing = [k for k, v in snap.valuation.items() if v is None]
    if missing:
        snap.notes.append("Unavailable from data source: " + ", ".join(missing))
    if bench_close is not None:
        snap.notes.append(f"Benchmark used for beta/correlation: {benchmark}")
    return snap
