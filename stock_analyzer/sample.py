"""Synthetic offline data so the pipeline can be tried without network access.

The numbers are generated, not real market data. The ticker is always DEMO
so a sample report can never be mistaken for a real company's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Snapshot, build_snapshot


def sample_snapshot(seed: int = 7) -> Snapshot:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=504)
    bench_rets = rng.normal(0.0004, 0.010, len(days))
    # stock = 1.15 x market + its own noise, so beta/correlation are meaningful
    rets = 1.15 * bench_rets + rng.normal(0.0002, 0.012, len(days))
    close = 100 * np.cumprod(1 + rets)
    bench = 400 * np.cumprod(1 + bench_rets)
    hist = pd.DataFrame({"Close": close}, index=days)

    info = {
        "longName": "Demo Corp (synthetic sample data)",
        "sector": "Technology",
        "industry": "Software",
        "currency": "USD",
        "marketCap": 85_000_000_000,
        "enterpriseValue": 82_000_000_000,
        "trailingPE": 31.2,
        "forwardPE": 25.4,
        "trailingPegRatio": 1.6,
        "priceToSalesTrailing12Months": 7.9,
        "priceToBook": 9.1,
        "enterpriseToEbitda": 21.5,
        "enterpriseToRevenue": 7.6,
        "profitMargins": 0.24,
        "operatingMargins": 0.29,
        "grossMargins": 0.71,
        "returnOnEquity": 0.33,
        "revenueGrowth": 0.14,
        "earningsGrowth": 0.19,
        "debtToEquity": 42.0,
        "currentRatio": 1.8,
        "freeCashflow": 3_100_000_000,
        "dividendYield": None,
        "beta": 1.15,
        "recommendationKey": "buy",
        "targetMeanPrice": float(close[-1] * 1.12),
        "numberOfAnalystOpinions": 28,
    }
    quarterly = [
        {"quarter_end": "2026-09-30", "revenue": 2.95e9, "net_income": 7.2e8, "diluted_eps": 1.21},
        {"quarter_end": "2026-06-30", "revenue": 2.81e9, "net_income": 6.7e8, "diluted_eps": 1.12},
        {"quarter_end": "2026-03-31", "revenue": 2.70e9, "net_income": 6.1e8, "diluted_eps": 1.03},
        {"quarter_end": "2025-12-31", "revenue": 2.64e9, "net_income": 6.3e8, "diluted_eps": 1.05},
    ]
    snap = build_snapshot("DEMO", hist, info, pd.Series(bench, index=days), quarterly)
    snap.notes.insert(0, "SYNTHETIC SAMPLE DATA for testing. Not a real company.")
    return snap
