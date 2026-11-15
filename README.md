# Stock Analyzer (Claude + yfinance)

A command line tool that turns any ticker into a consistent, structured research report.
Python computes the numbers; Claude interprets them using one fixed prompt template, so
every report has the same sections and scoring.

```
yfinance ──► data.py (trend, valuation, risk math) ──► JSON snapshot
                                                          │
                       prompt.py (fixed report template) ─┤
                                                          ▼
                                     analyzer.py ──► Claude ──► Markdown report
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."     # from console.anthropic.com
```

The key is read from the environment and never stored in code. (If you use the
`ant` CLI, `ant auth login` works too.)

## Usage

```bash
python -m stock_analyzer AAPL                         # one report, streamed to the terminal
python -m stock_analyzer AAPL MSFT NVDA --save        # several, saved to reports/ as .md + .json
python -m stock_analyzer TSLA --focus "Is the valuation justified by growth?"
python -m stock_analyzer AAPL --effort high           # deeper analysis (slower, more tokens)
python -m stock_analyzer AAPL --data-only             # just the computed numbers, no Claude call
python -m stock_analyzer AAPL --show-prompt           # see exactly what Claude receives
python -m stock_analyzer --sample                     # offline demo on synthetic data (ticker DEMO)
```

Options: `--period` (price history, default `2y`), `--benchmark` (for beta, default `SPY`),
`--model` (default `claude-opus-5-5`), `--effort` (`low` to `max`, default `medium`).

## What goes into each report

| Area | Metrics computed locally |
|---|---|
| Trend | 1m/3m/6m/1y returns, 50 and 200 day averages, golden cross, 52 week range, last 10 closes |
| Valuation | Trailing and forward P/E, PEG, EV/EBITDA, EV/revenue, P/S, P/B, analyst target and upside |
| Fundamentals | Last 4 quarters of revenue, net income, EPS; margins, ROE, growth, debt to equity, current ratio, FCF |
| Risk | Annualized and 30 day volatility, beta (reported and computed vs benchmark), max and current drawdown, Sharpe, Sortino, 95% daily VaR |

The report sections are always: Summary, Price trend, Valuation, Fundamentals, Risk assessment
(with a Low/Moderate/Elevated/High rating), a 1 to 5 Scorecard, and Bottom line. Edit
`SYSTEM_PROMPT` in `stock_analyzer/prompt.py` to change the template for every ticker at once.

## Design choices

- **Structured data, not raw tables.** Claude gets a compact JSON snapshot with clear units,
  and is told to say "unavailable" instead of guessing when a field is missing.
- **Current model and API features.** Uses `claude-opus-5-5` with adaptive thinking, an effort
  setting, streaming, and prompt caching on the fixed system prompt (cheaper repeat runs).
  Server-side refusal fallback is enabled so a declined request is retried on another model.
- **Not financial advice.** The template forbids buy/sell/hold instructions.

## Tests

```bash
pip install pytest
python -m pytest tests
```

Tests run fully offline: they check the metric math on synthetic prices and verify the
Claude request shape against a mocked HTTP transport.

## Ideas for next steps

Web dashboard with charts (Flask or Streamlit), peer comparison across a sector, portfolio
tracking in SQLite, price and earnings date alerts, and news headline sentiment.
