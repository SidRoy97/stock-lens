# stock-lens

**Next-day S&P 500 direction prediction with an honest out-of-sample story — and a chatbot that explains itself in plain English.**

stock-lens is an end-to-end machine learning system built for CS6140 (Machine Learning, Northeastern University). It trains tabular and deep sequence models on 2010–2016 S&P 500 data to classify each stock's next trading day as **Up** (>+1%), **Down** (<−1%), or **Neutral**, measures how those models age on 2017–2026 market data, and wraps everything in a natural-language chatbot that answers questions about predictions, confidence, feature attribution, sector rankings — on the historical dataset or on **live market data** fetched at question time.

**Authors:** Siddhartha Roy · Kaylee Faherty · Ran Fukazawa

> Educational project. Nothing here is financial advice — every answer the chatbot gives says so too.

---

## Ask it anything

```
you> Will Apple go up tomorrow?
you> What is the confidence level for MSFT's prediction?
you> Should I sell or hold my Apple stock?
you> What are the top features affecting the XOM prediction?
you> Which sector should I buy into for gains next week?
you> Which stocks have the highest predicted probability of a >1% gain tomorrow?
you> Is now a good time to buy into the Energy sector?
you> Will TSLA go up tomorrow? live
```

The last one is the party trick: the 2010–2016 dataset has no TSLA, but adding the word **`live`** routes any current US ticker through a live-inference path — ~250 days of yfinance data pushed through the *exact same* feature code used in training (SPY and sector-ETF proxies stand in for cross-sectional averages; unavailable fundamentals are imputed with training means). Same models, same features, today's data.

Eight intents are parsed by a rule-based NLP layer (ticker/company-name resolution, dates, sectors, horizons, model choice): prediction, confidence, buy/sell/hold advisory framing, feature-level explanation, top-stock screening, sector ranking, sector assessment, and multi-day horizon questions — which are answered honestly ("this model's trained horizon is one day").

---

## Results

Pooled training, classification heads, strictly chronological splits (macro-F1, 3-class, random baseline ≈ 0.33):

| Model | Validation | Test |
|---|---|---|
| Random forest | 0.38 | — |
| XGBoost | 0.39 | 0.40 |
| LSTM | 0.44 | — |
| GRU | 0.45 | — |
| TCN | 0.45 | — |
| Transformer | 0.46 | — |
| **CNN (1D)** | **0.46** | **0.47** |

The held-out test period is touched **once**, by the validation winner — evaluating every candidate on test and reporting the best would be test-set selection bias. Two ablations validated the design: classification heads beat return-regression heads on every architecture (TCN: 0.45 vs 0.15), and pooled cross-ticker training beats per-ticker training roughly **2×** (0.44–0.46 vs 0.22–0.25 over 40 stocks).

**The central finding — regime drift:** evaluated frozen on 2017–2025 yfinance data, the CNN's 0.468 collapses to an average of **0.353** (RF: 0.357) — near the random baseline *from the very first out-of-era year*. Degradation is not gradual erosion but an immediate consequence of leaving the training distribution. For price-only models, the dominant failure mode is not architecture but distribution shift.

---

## Pipeline

| Stage | `python main.py --stage N` | What it does |
|---|---|---|
| 1 | data loading & EDA | Kaggle NYSE prices/fundamentals/securities → long format, exploratory plots |
| 2 (+2b) | features & labels | RSI, MACD, MAs, volume ratio, lags, multi-horizon returns, market/sector-relative strength; ±1% 3-class labels; horizon studies → `master_enhanced.csv` |
| 3 | tabular models | class-balanced Random Forest + XGBoost; importances later power chatbot explanations |
| 4 | sequence models | CNN/LSTM/GRU/TCN/Transformer bake-off over sliding windows, chronological val/test |
| 5 | **chatbot** | CLI or Gradio web UI; add `--chat-mode gradio` |
| 6 | out-of-sample study | frozen models vs 2017–2026 yfinance data, year by year |

All rolling features use only past data; every split is chronological; scalers fit on training rows only. `observations/` holds every generated figure and `run_log.txt` records every reported metric.

## Repository layout

```
code/            flat Python modules (main.py, config.py, features.py,
                 sequence_models.py, predictors.py, nlp_parser.py,
                 chatbot.py, live_features.py, oos_evaluation.py, ...)
models/          trained artifacts (committed — run the chatbot without retraining)
observations/    all figures + run_log.txt
data/            Kaggle CSVs + generated master_enhanced.csv (not committed — see setup)
```

## Setup

```bash
git clone https://github.com/SidRoy97/stock-lens
cd stock-lens
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

1. **Data** (not committed for size): download the Kaggle "New York Stock Exchange" dataset
   (`dgawlik/nyse`) and place `prices-split-adjusted.csv`, `fundamentals.csv`,
   `securities.csv` in `data/`.
2. Point the code at this directory:
   `export STOCK_LENS_BASE=$(pwd)`
3. Rebuild the feature table (models are already committed, so no training needed):
   `cd code && python main.py --stage 1 && python main.py --stage 2`
4. Chat: `python main.py --stage 5` (CLI) or
   `python main.py --stage 5 --chat-mode gradio` (web UI, shareable link).
   Live mode needs internet for yfinance.
5. Optional: retrain everything (`--stage 3`, `--stage 4`) or reproduce the
   drift study (`--stage 6`).

Optional: set `ANTHROPIC_API_KEY` to have the chatbot add plain-English narrative explanations of model outputs.

## Data sources

Kaggle NYSE dataset (D. Gawlik) — 501 tickers, ~851k daily rows, 2010–2016, with fundamentals and GICS sectors · Yahoo Finance via `yfinance` — out-of-sample evaluation and live inference · SPDR sector ETFs — live sector-relative proxies.

## What we'd want you to take away

The interesting number here isn't 0.47 — it's the drop to 0.35. This project treats evaluation design as the contribution: leak-free chronological splits, a single-touch test protocol, and an out-of-sample study that reports the models getting *worse* honestly. The chatbot and live-inference layer then demonstrate the deployment half: feature parity between training and serving, explicit assumptions where live data has gaps, and a user interface that states its own limitations in every answer.
