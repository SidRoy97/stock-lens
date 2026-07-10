"""thorough offline test harness for the stock-lens chatbot fixes

stubs the models and data so routing, counts, typing, and formatting can
be verified without training artifacts — run: python test_chatbot_full.py
"""
import sys, types
import numpy as np
import pandas as pd

# ---- stub config + helpers so imports work anywhere ----
cfg = types.ModuleType("config"); cfg.DATA_PATH = "/tmp/sl_test"
sys.modules["config"] = cfg
helpers = types.ModuleType("helpers")
helpers.log = lambda *a, **k: None
helpers.section = lambda *a, **k: None
sys.modules["helpers"] = helpers

import os, importlib.util
os.makedirs("/tmp/sl_test", exist_ok=True)

# ---- build a securities.csv with a KNOWN stale ticker (TWX) ----
securities = pd.DataFrame({
    "Ticker symbol": ["HBI", "CERN", "SWK", "SNA", "XOM", "CVX", "AAPL",
                      "MSFT", "JPM", "BAC", "TWX", "PFE", "JNJ", "DUK"],
    "Security": ["Hanesbrands Inc", "Cerner Corp", "Stanley Black Decker",
                 "Snap-on Inc", "Exxon Mobil Corp", "Chevron Corp",
                 "Apple Inc", "Microsoft Corp", "JPMorgan Chase",
                 "Bank of America", "Time Warner Inc", "Pfizer Inc",
                 "Johnson & Johnson", "Duke Energy"],
    "GICS Sector": ["Consumer Discretionary", "Health Care",
                    "Consumer Discretionary", "Consumer Discretionary",
                    "Energy", "Energy", "Information Technology",
                    "Information Technology", "Financials", "Financials",
                    "Consumer Discretionary", "Health Care", "Health Care",
                    "Utilities"]})
securities.to_csv("/tmp/sl_test/securities.csv", index=False)

# ---- load the REAL parser under test ----
spec = importlib.util.spec_from_file_location("nlp_parser", "nlp_parser.py")
nlp = importlib.util.module_from_spec(spec); spec.loader.exec_module(nlp)
sys.modules["nlp_parser"] = nlp

tickers, name_map = nlp.load_ticker_names()
sector_map, valid_sectors = nlp.load_sector_map()
LATEST = pd.Timestamp("2016-12-22")

# ---- stub predictors module (chatbot imports from it) ----
pred_mod = types.ModuleType("predictors")
class FakeRFModel:
    feature_importances_ = np.array([0.5, 0.3, 0.2])
    def predict_proba(self, X):
        rng = np.random.default_rng(42)
        p = rng.uniform(0.2, 0.55, (len(X), 3)); p /= p.sum(1, keepdims=True)
        return p
class Enc: classes_ = np.array(["Down", "Neutral", "Up"])
class Scl:
    mean_ = np.zeros(3); scale_ = np.ones(3)
    def transform(self, X): return np.asarray(X, float)
class Imp:
    def transform(self, X): return np.asarray(X, float)
RF = {"model": FakeRFModel(), "scaler": Scl(), "imputer": Imp(),
      "feature_cols": ["f1", "f2", "f3"], "label_encoder": Enc()}
pred_mod.load_seq_predictor = lambda: None
pred_mod.load_rf_predictor = lambda: RF
pred_mod.predict_one = lambda t, d, m, s, r, data: {
    "ticker": t, "date": str(d), "model": m, "prediction": "Neutral",
    "confidence": 0.43}
pred_mod.format_answer = lambda p: (f"{p['ticker']}: {p['prediction']} "
    f"{p['confidence']:.0%}" if "error" not in p else p["error"])
sys.modules["predictors"] = pred_mod
lf = types.ModuleType("live_features")
lf.predict_live = lambda *a, **k: {"error": "live disabled in test"}
sys.modules["live_features"] = lf

# ---- build master_enhanced.csv: 14 tickers, some UNSCOREABLE (NaN feat) ----
rows = []
syms = list(securities["Ticker symbol"])
for i, s in enumerate(syms):
    # make 2 tickers unscoreable to prove rankings stay complete anyway
    f1 = np.nan if s in ("DUK", "JNJ") else float(i + 1)
    rows.append({"date": LATEST, "symbol": s,
                 "f1": f1, "f2": float(i), "f3": float(i * 2)})
master = pd.DataFrame(rows)
master.to_csv("/tmp/sl_test/master_enhanced.csv", index=False)

# ---- load the REAL chatbot, but capture its inner answer() via a shim ----
spec2 = importlib.util.spec_from_file_location("chatbot", "chatbot.py")
chatbot = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(chatbot)

# stage_5 builds closures then enters a loop; we replicate its setup to get answer()
# by monkeypatching input to feed questions and capture prints.
QUESTIONS = [
    ("Which stocks have the highest predicted probability of a gain tomorrow?",
     ["top_movers"]),
    ("Which sector should I buy into for gains next week?",
     ["sector_rank"]),
    ("Is now a good time to buy into the Energy sector?",
     ["sector_check", "Energy"]),
    ("show me the top 5 stocks", ["top_movers"]),
    ("what are the top 3 sectors", ["sector_rank"]),
    ("will apple go up tomorrow", ["predict", "AAPL"]),
    ("how confident are you about MSFT", ["confidence", "MSFT"]),
    ("why is XOM predicted up", ["explain", "XOM"]),
    ("should I sell or hold my apple stock", ["advise", "AAPL"]),
    ("how is the financial sector doing", ["sector_check", "Financials"]),
    ("how are healthcare stocks looking", ["sector_check", "Health Care"]),
    ("which sector looks best right about this time", ["sector_rank"]),
]

captured = []
answers = []
orig_input = __builtins__.input if hasattr(__builtins__, "input") else input
feed = iter([qq for qq, _ in QUESTIONS] + ["quit"])
def fake_input(prompt=""):
    return next(feed)
def fake_print(*a, **k):
    if a and isinstance(a[0], str) and a[0].startswith("\nbot> "):
        answers.append(a[0][6:])
import builtins
builtins.input = fake_input
_op = builtins.print
builtins.print = fake_print
try:
    chatbot.stage_5_chatbot(mode="cli")
finally:
    builtins.print = _op
    builtins.input = orig_input

# ---- also directly test the parser routing for assertions ----
def route(text):
    return nlp.parse_query(text, tickers, name_map, LATEST, "sequence",
                           valid_sectors)

print = _op
print("=" * 68)
print("ROUTING ASSERTIONS")
print("=" * 68)
fails = 0
for text, expect in QUESTIONS:
    q = route(text)
    ok = q["intent"] == expect[0]
    if len(expect) > 1:
        ok = ok and (q["ticker"] == expect[1] or q["sector"] == expect[1])
    flag = "OK " if ok else "FAIL"
    if not ok: fails += 1
    extra = f"ticker={q['ticker']} sector={q['sector']} top_n={q['top_n']}"
    print(f"  [{flag}] {text[:52]:<52} -> {q['intent']:<12} {extra}")

print()
print("=" * 68)
print("ACTUAL BOT ANSWERS (the three screenshot cases + more)")
print("=" * 68)
for (text, _), ans in zip(QUESTIONS, answers):
    print(f"\nyou> {text}")
    print(f"bot> {ans}")

print()
print("=" * 68)
print("REGRESSION CHECKS ON THE THREE ORIGINAL BUGS")
print("=" * 68)

# Bug1: top movers returns 5 (12 scoreable, want 5)
tm = [a for (t, _), a in zip(QUESTIONS, answers)
      if t.startswith("Which stocks")][0]
n_listed = sum(1 for ln in tm.splitlines() if ln.strip()[:1].isdigit())
print(f"  top movers listed {n_listed} stocks (want 5): "
      f"{'PASS' if n_listed == 5 else 'FAIL'}")
fails += n_listed != 5

# Bug2a: no decimal counts anywhere
sr = [a for (t, _), a in zip(QUESTIONS, answers)
      if t.startswith("Which sector should")][0]
has_decimal = ".0 stocks" in sr
print(f"  sector ranking decimal counts present: "
      f"{'FAIL' if has_decimal else 'PASS'} "
      f"({'found .0' if has_decimal else 'integers'})")
fails += has_decimal

# Bug2b: 3+ sectors available get shown (we have 5 distinct scoreable sectors)
n_sec = sum(1 for ln in sr.splitlines() if ln.strip()[:1].isdigit())
print(f"  sector ranking showed {n_sec} sectors (want 3): "
      f"{'PASS' if n_sec == 3 else 'FAIL'}")
fails += n_sec != 3

# Bug3: energy question does NOT mention TWX and IS a sector assessment
en = [a for (t, _), a in zip(QUESTIONS, answers)
      if "Energy sector" in t][0]
twx_leak = "TWX" in en
is_sector = "Energy" in en and ("sector" in en.lower() or "ranking" in en.lower())
print(f"  energy question leaked TWX: {'FAIL' if twx_leak else 'PASS'}")
print(f"  energy question is sector assessment: "
      f"{'PASS' if is_sector else 'FAIL'}")
fails += twx_leak or not is_sector

print()
print("=" * 68)
print(f"RESULT: {'ALL PASS' if fails == 0 else str(fails) + ' FAILURES'}")
print("=" * 68)
