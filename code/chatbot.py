"""running the chatbot that understands natural language questions"""

import os
import numpy as np
import pandas as pd
from config import DATA_PATH
from helpers import log, section
from predictors import (load_seq_predictor, load_rf_predictor,
                        predict_one, format_answer)
from nlp_parser import load_ticker_names, load_sector_map, parse_query
from live_features import predict_live

DISCLAIMER = "\n\nEducational output only — not financial advice."


def stage_5_chatbot(mode="cli"):
    section("STAGE 5 — CHATBOT (natural language)")

    # loading models, data, and the lookup tables
    seq_pred = load_seq_predictor()
    rf_pred = load_rf_predictor()
    log(f"sequence model loaded: {seq_pred is not None}")
    log(f"random forest loaded: {rf_pred is not None}")
    if seq_pred is None and rf_pred is None:
        log("no models found — run stage 3 and stage 4 first")
        return
    data = pd.read_csv(os.path.join(DATA_PATH, "master_enhanced.csv"),
                       parse_dates=["date"])
    tickers, name_map = load_ticker_names()
    sector_map, valid_sectors = load_sector_map()
    latest_date = data["date"].max()
    default_model = "sequence" if seq_pred is not None else "random_forest"
    cache = {}

    def latest_scores():
        # scoring every ticker's latest row once, then reusing the table
        if "scores" in cache:
            return cache["scores"]
        if rf_pred is None:
            return None
        latest = data[data["date"] == latest_date].drop_duplicates("symbol")
        # dropping rows the model cannot score cleanly so every downstream
        # ranking has complete, honest data instead of a silently short list
        feat = rf_pred["feature_cols"]
        latest = latest.dropna(subset=feat)
        if latest.empty:
            cache["scores"] = None
            return None
        X = rf_pred["scaler"].transform(
            rf_pred["imputer"].transform(latest[feat]))
        probs = rf_pred["model"].predict_proba(X)
        classes = list(rf_pred["label_encoder"].classes_)
        out = pd.DataFrame({
            "ticker": latest["symbol"].values,
            "p_up": probs[:, classes.index("Up")],
            "p_down": probs[:, classes.index("Down")]})
        out["sector"] = out["ticker"].map(sector_map)
        out["net"] = out["p_up"] - out["p_down"]
        cache["scores"] = out
        return out

    def sector_table():
        # aggregating per-sector optimism from the per-ticker scores
        scores = latest_scores()
        if scores is None:
            return None
        table = scores.dropna(subset=["sector"]).groupby("sector") \
            .agg(net=("net", "mean"), p_up=("p_up", "mean"),
                 n=("ticker", "count")).sort_values("net", ascending=False)
        if table.empty:
            return None
        # counts are whole stocks — keep them integer so they never render
        # as 1.0 / 3.0 after any float-introducing aggregation upstream
        table["n"] = table["n"].astype(int)
        return table

    def get_pred(q):
        # choosing live yfinance data or the 2010-2016 dataset per the query
        if q.get("live"):
            return predict_live(q["ticker"], q["model"], seq_pred, rf_pred)
        return predict_one(q["ticker"], q["date"], q["model"],
                           seq_pred, rf_pred, data)

    def answer_predict(q):
        return format_answer(get_pred(q))

    def answer_confidence(q):
        pred = get_pred(q)
        if "error" in pred:
            return format_answer(pred)
        return (f"The {pred['model']} model predicts {pred['ticker']} will go "
                f"{pred['prediction']} on the next trading day after "
                f"{pred['date']}, with {pred['confidence']:.0%} confidence. "
                f"Anything near 33% is a coin-flip across the three classes; "
                f"{pred['confidence']:.0%} means the model sees a "
                f"{'clear' if pred['confidence'] > 0.5 else 'mild'} edge."
                + DISCLAIMER)

    def answer_advise(q):
        pred = get_pred(q)
        if "error" in pred:
            return format_answer(pred)
        lean = {"Up": "leans positive — historically consistent with "
                      "holding rather than selling",
                "Down": "leans negative — historically consistent with "
                        "trimming or tightening a stop",
                "Neutral": "sees no clear edge either way — consistent "
                           "with simply holding"}[pred["prediction"]]
        return (f"I can't tell you what to do with your money, but here is "
                f"what the model sees for {pred['ticker']}: the next-day "
                f"signal is {pred['prediction']} at {pred['confidence']:.0%} "
                f"confidence, which {lean}. Remember this model predicts one "
                f"day ahead from 2010-2016 patterns — position decisions "
                f"should weigh far more than that." + DISCLAIMER)

    def answer_horizon(q):
        pred = get_pred(q)
        if "error" in pred:
            return format_answer(pred)
        return (f"Honest limitation first: this model is trained to predict "
                f"exactly one trading day ahead, so a {q['horizon_days']}-day "
                f"view is beyond what it can credibly forecast. Its next-day "
                f"signal for {pred['ticker']} is {pred['prediction']} at "
                f"{pred['confidence']:.0%} confidence. For multi-day "
                f"performance, treat the day-ahead signal as a weak prior, "
                f"not a path forecast." + DISCLAIMER)

    def answer_explain(q):
        if rf_pred is None:
            return "Explanations need the random forest model loaded."
        pred = predict_one(q["ticker"], q["date"], "random_forest",
                           seq_pred, rf_pred, data)
        if "error" in pred:
            return format_answer(pred)
        cols = rf_pred["feature_cols"]
        imps = rf_pred["model"].feature_importances_
        top = sorted(zip(cols, imps), key=lambda p: -p[1])[:6]
        sub = data[data["symbol"] == q["ticker"]]
        row = sub[sub["date"] <= pd.Timestamp(q["date"])].iloc[-1]
        means = dict(zip(cols, rf_pred["scaler"].mean_))
        scales = dict(zip(cols, rf_pred["scaler"].scale_))
        lines = []
        for c, imp in top:
            val = row.get(c)
            if pd.isna(val):
                continue
            z = (val - means[c]) / scales[c] if scales[c] else 0
            direction = "above" if z > 0 else "below"
            lines.append(f"  - {c}: {val:.3g} ({abs(z):.1f} std devs "
                         f"{direction} the dataset average, importance "
                         f"{imp:.1%})")
        note = ""
        if q["model"] == "sequence":
            note = ("\n(Feature attribution comes from the random forest, "
                    "which learns from the same features — the CNN itself "
                    "is not directly interpretable.)")
        return (f"{q['ticker']} is predicted {pred['prediction']} at "
                f"{pred['confidence']:.0%} confidence. The model's most "
                f"influential features and this stock's current readings:\n"
                + "\n".join(lines) + note + DISCLAIMER)

    def answer_top_movers(q):
        scores = latest_scores()
        if scores is None:
            return "Ranking needs the random forest model loaded."
        want = q.get("top_n") or 5
        ranked = scores.sort_values("p_up", ascending=False)
        top = ranked.head(want)
        lines = [f"  {i + 1}. {r.ticker} ({r.sector}) — P(>1% gain "
                 f"tomorrow) = {r.p_up:.0%}"
                 for i, r in enumerate(top.itertuples())]
        note = ""
        if len(top) < want:
            note = (f"\n(Only {len(top)} stocks had complete data to score "
                    f"today, so I'm showing those.)")
        return (f"Top {len(top)} by model-estimated probability of an Up day "
                f"(defined as a >1% gain) after {latest_date.date()}:\n"
                + "\n".join(lines) + note + DISCLAIMER)

    def answer_sector_rank(q):
        table = sector_table()
        if table is None:
            return "Sector ranking needs the random forest model loaded."
        want = q.get("top_n") or 3
        shown = table.head(want)
        lines = [f"  {i + 1}. {name} — avg P(Up) {row.p_up:.0%}, net "
                 f"optimism {row.net:+.2f} across {int(row.n)} stocks"
                 for i, (name, row) in enumerate(shown.iterrows())]
        note = ""
        if len(shown) < want:
            note = f"\n(Only {len(table)} sectors were scoreable today.)"
        return (f"Top {len(shown)} sectors by the model's average next-day "
                f"optimism as of {latest_date.date()}:\n" + "\n".join(lines)
                + note
                + f"\n\nNote the model's horizon is one day, so 'next week' "
                  f"is an extrapolation." + DISCLAIMER)

    def answer_sector_check(q):
        table = sector_table()
        if table is None or q["sector"] not in table.index:
            return f"I couldn't score the {q['sector']} sector."
        row = table.loc[q["sector"]]
        rank = int(table.index.get_loc(q["sector"])) + 1
        mood = ("relatively optimistic" if rank <= 3 else
                "middling" if rank <= 7 else "relatively pessimistic")
        return (f"For {q['sector']}: the model's average P(Up) across its "
                f"{int(row.n)} stocks is {row.p_up:.0%}, ranking it "
                f"{rank} of {len(table)} sectors — {mood} by next-day "
                f"signal as of {latest_date.date()}. Whether that makes now "
                f"'a good time' depends on horizon and risk appetite the "
                f"model cannot see." + DISCLAIMER)

    handlers = {"predict": answer_predict, "confidence": answer_confidence,
                "advise": answer_advise, "explain": answer_explain,
                "top_movers": answer_top_movers,
                "sector_rank": answer_sector_rank,
                "sector_check": answer_sector_check}

    def answer(question):
        # parsing the question and routing it to the right handler
        q = parse_query(question, tickers, name_map, latest_date,
                        default_model, valid_sectors)
        if q["horizon_days"] and q["intent"] in ("predict", "advise") \
                and q["ticker"]:
            return answer_horizon(q)
        needs_ticker = q["intent"] in ("predict", "confidence", "advise",
                                       "explain")
        if q.get("live") and q["intent"] in ("top_movers",
                                             "sector_rank", "sector_check"):
            return ("Live mode currently supports single-ticker questions; "
                    "sector rankings and stock screens run on the "
                    "2010-2016 dataset. Drop the word 'live' for those.")
        if needs_ticker and q["ticker"] is None:
            return ("Sorry, that stock isn't in my dataset (S&P 500 "
                    "companies, 2010-2016 — no TSLA or AMZN, for example), "
                    "or the question needs a ticker. Try: 'Will Apple go "
                    "up tomorrow?', 'Which sector should I buy into?', or "
                    "add the word 'live' to ask about any current US "
                    "ticker, e.g. 'Will TSLA go up tomorrow? live'")
        return handlers[q["intent"]](q)

    # launching the polished gradio web interface when requested
    if mode in ("gradio", "both"):
        try:
            from gradio_ui import build_interface
            demo = build_interface(answer)
            log("launching gradio...")
            demo.launch(share=True, **getattr(demo, "_launch_extras", {}))
            return
        except ImportError:
            log("gradio missing — falling back to CLI")

    # running the command-line chat loop
    log("\nchat ready — ask naturally, e.g. 'Which sector should I buy "
        "into?' or 'Why is XOM predicted up?'")
    log("type 'quit' to exit")
    while True:
        try:
            raw = input("\nyou> ").strip()
        except EOFError:
            break
        if raw.lower() in ("quit", "exit", "q"):
            break
        if raw:
            print("\nbot> " + answer(raw))
    log("chat session ended")
