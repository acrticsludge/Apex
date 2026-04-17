# RL Trading Agent

This package is a standalone reinforcement-learning trading system built around:

- `yfinance` for 10 years of daily OHLCV data
- `pandas-ta` for indicator engineering
- Finnhub company news plus VADER sentiment scoring
- A custom `Gymnasium` trading environment
- `Stable-Baselines3` PPO with `MlpPolicy`
- `FastAPI` for external bot integration

It does **not** integrate into `apex_dashboard.py` yet. The only dashboard-aware behavior is that `trading_agent/config.py` reads the dashboard watchlists and config literals as defaults when that file exists.

The package is now also set up for Google Colab:

- it defaults to your combined India + US watchlists
- it can store artifacts in Google Drive instead of temporary notebook storage
- it saves periodic PPO checkpoints for resume-friendly training sessions

## Folder Layout

```text
trading_agent/
├── data/
│   ├── data_fetcher.py
│   ├── indicator_engine.py
│   └── sentiment_engine.py
├── environment/
│   └── trading_env.py
├── agent/
│   ├── train.py
│   ├── evaluate.py
│   └── model/
├── integration/
│   └── bot_bridge.py
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:

```bash
pip install -r trading_agent/requirements.txt
```

2. Create a `.env` file either in the repo root or inside `trading_agent/`:

```env
FINNHUB_API_KEY=your_finnhub_key_here
ENABLE_SENTIMENT=true
RL_MARKET_UNIVERSE=all
```

3. Optional quick-start note:

- If you want a fast smoke test before pulling news sentiment, set `ENABLE_SENTIMENT=false`.
- The first sentiment-enabled run is the slowest because the Finnhub news series is cached locally under `trading_agent/data/cache/sentiment/`.

## Train

Run PPO training with the config defaults:

```bash
python -m trading_agent.main --mode train
```

By default, the training universe is your combined dashboard-style watchlist:

- India: `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ICICIBANK.NS`, `HINDUNILVR.NS`, `ITC.NS`, `SBIN.NS`, `BHARTIARTL.NS`, `KOTAKBANK.NS`, `LT.NS`, `AXISBANK.NS`, `MARUTI.NS`, `TITAN.NS`, `WIPRO.NS`, `SUNPHARMA.NS`
- US: `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `AMZN`, `META`, `TSLA`, `AMD`, `NFLX`, `ORCL`, `INTC`, `CRM`, `UBER`, `SHOP`, `PYPL`, `PLTR`

Train on a smaller custom list first:

```bash
python -m trading_agent.main --mode train --tickers AAPL MSFT NVDA
```

What training does:

- downloads 10 years of daily OHLCV data with `yfinance`
- adds RSI, MACD, Bollinger Bands, EMA20, EMA50, ATR14, OBV
- fetches Finnhub company news, scores it into a `[-1, +1]` sentiment feature, and caches it
- splits chronologically into 80% train / 10% validation / 10% test
- fits a `MinMaxScaler` on the train split only
- rewards the policy for absolute returns plus excess return versus buy-and-hold
- penalizes repeated flat exposure in rising markets and invalid over-trading actions
- trains PPO for `500000` timesteps
- saves the best checkpoint by a validation selection score that combines Sharpe, benchmark outperformance, drawdown, and trade coverage
- saves periodic resume checkpoints during training

Saved outputs land in `trading_agent/agent/model/`:

- `best_model.zip`
- `latest_model.zip`
- `scaler.joblib`
- `feature_columns.json`
- `dataset_metadata.json`
- `training_summary.json`
- `validation_metrics.json`
- `checkpoints/ppo_checkpoint_*.zip`

## Evaluate

Run a standalone backtest after training:

```bash
python -m trading_agent.main --mode evaluate
```

This saves:

- `evaluation_metrics.json`
- `cumulative_return_comparison.png`

Reported metrics:

- Cumulative Return
- Excess Return vs Buy-and-Hold
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Buy-and-Hold benchmark comparison
- Action mix, zero-trade ticker count, average holding period, and skipped ticker diagnostics

## Serve

Start the FastAPI bridge:

```bash
python -m trading_agent.main --mode serve
```

Available endpoints:

- `POST /predict`
- `GET /status`
- `POST /retrain`

Example `POST /predict` payload:

```json
{
  "observation": [0.12, 0.55, 0.34, 0.49, 0.77, 0.41, 0.58, 0.53, 0.51, 0.39, 0.46, 0.63, 0.57, 0.61, 0.44, 0.80, 0.50]
}
```

Important:

- The bridge expects the observation vector to already be normalized with the saved scaler and ordered exactly like `feature_columns.json`.
- This keeps the bridge simple for later integration with your existing bot.

## Full Pipeline

Run training, then evaluation, then start serving:

```bash
python -m trading_agent.main --mode full --tickers AAPL MSFT NVDA
```

## Google Colab

This package is Colab-friendly and includes a dedicated launcher at `trading_agent/colab_runner.py`.

Recommended Colab flow:

1. Upload or clone this repo into Colab.
2. Switch Colab runtime to GPU.
3. Install dependencies.
4. Add your Finnhub key.
5. Train with Drive-backed checkpointing.

Example notebook cells:

```python
!pip install -q -r trading_agent/requirements.txt
```

```python
import os
os.environ["FINNHUB_API_KEY"] = "your_finnhub_key_here"
os.environ["ENABLE_SENTIMENT"] = "true"
os.environ["RL_MARKET_UNIVERSE"] = "all"
```

```python
!python -m trading_agent.colab_runner --resume
```

What `trading_agent.colab_runner` does:

- mounts Google Drive automatically when running in Colab
- stores artifacts under `/content/drive/MyDrive/trading_agent_runtime/`
- trains on the combined India + US watchlists by default
- saves checkpoints so you can rerun with `--resume`
- runs evaluation after training unless you pass `--skip-evaluate`

Useful Colab commands:

```python
!python -m trading_agent.colab_runner --resume --timesteps 200000
!python -m trading_agent.colab_runner --resume --skip-evaluate
!python -m trading_agent.colab_runner --tickers RELIANCE.NS TCS.NS HDFCBANK.NS
```

If you prefer the generic CLI instead of the Colab helper, this also works:

```python
!python -m trading_agent.main --mode train --resume --storage-dir /content/drive/MyDrive/trading_agent_runtime
```

Use a GPU runtime for faster PPO training. The data pipeline itself is still mostly CPU/network bound, so the biggest speedup usually appears once training starts. On free Colab, long runs can still disconnect, which is why resume checkpoints are now part of the default workflow.
