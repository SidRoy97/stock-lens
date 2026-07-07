"""parsing natural language questions into ticker, date, model, and intent"""

import os
import re
import pandas as pd
from config import DATA_PATH

SECTOR_SYNONYMS = {
    "tech": "Information Technology",
    "technology": "Information Technology",
    "it sector": "Information Technology",
    "energy": "Energy",
    "oil": "Energy",
    "financial": "Financials",
    "financials": "Financials",
    "banks": "Financials",
    "banking": "Financials",
    "health": "Health Care",
    "healthcare": "Health Care",
    "health care": "Health Care",
    "pharma": "Health Care",
    "utilities": "Utilities",
    "industrial": "Industrials",
    "industrials": "Industrials",
    "materials": "Materials",
    "real estate": "Real Estate",
    "telecom": "Telecommunications Services",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "staples": "Consumer Staples",
}


def load_ticker_names():
    # building ticker and company-name lookup tables from securities.csv
    sec = pd.read_csv(os.path.join(DATA_PATH, "securities.csv"))
    tickers = set(sec["Ticker symbol"].str.upper())
    name_map = {}
    for _, row in sec.iterrows():
        full = str(row["Security"]).lower()
        name_map[full] = row["Ticker symbol"]
        first = full.split()[0]
        # mapping the distinctive first word of each company name to its ticker
        if len(first) > 3 and first not in ("the", "first", "general",
                                            "american", "united", "national"):
            name_map.setdefault(first, row["Ticker symbol"])
    return tickers, name_map


def load_sector_map():
    # mapping each ticker to its gics sector for sector-level questions
    sec = pd.read_csv(os.path.join(DATA_PATH, "securities.csv"))
    sector_map = dict(zip(sec["Ticker symbol"].str.upper(),
                          sec["GICS Sector"]))
    return sector_map, set(sec["GICS Sector"].unique())


def parse_date(text, latest_date):
    # extracting a date from the question, defaulting to the latest available
    text = text.lower()
    iso = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if iso:
        return iso.group()
    us = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if us:
        m, d, y = us.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"
    month_pat = (r"(january|february|march|april|may|june|july|august|"
                 r"september|october|november|december)\s+(\d{1,2}),?\s*(\d{4})")
    named = re.search(month_pat, text)
    if named:
        try:
            return str(pd.to_datetime(named.group()).date())
        except Exception:
            pass
    # treating today, tomorrow, latest, and now as the newest available day
    return str(pd.Timestamp(latest_date).date())


def parse_model(text):
    # detecting which model the person wants from casual phrasing
    text = text.lower()
    if any(w in text for w in ("forest", "tree", " rf", "tabular", "simple")):
        return "random_forest"
    if any(w in text for w in ("cnn", "sequence", "neural", "deep", "lstm")):
        return "sequence"
    return None


def parse_ticker(text, tickers, name_map):
    # finding a ticker symbol or company name anywhere in the question
    stopwords = {"I", "A", "UP", "DOWN", "ON", "IN", "AT", "TO", "IS", "IT",
                 "GO", "DO", "BE", "OR", "SO", "USE", "THE", "CNN", "RF"}
    for token in re.findall(r"\b[A-Z][A-Z0-9.\-]{0,5}\b", text):
        if token in tickers and token not in stopwords:
            return token
    lowered = text.lower()
    # preferring the longest matching company name to avoid generic collisions
    matches = [(name, tk) for name, tk in name_map.items() if name in lowered]
    if matches:
        return max(matches, key=lambda m: len(m[0]))[1]
    return None


def parse_sector(text, valid_sectors):
    # matching sector synonyms, preferring the longest phrase found
    lowered = text.lower()
    hits = [(syn, canon) for syn, canon in SECTOR_SYNONYMS.items()
            if syn in lowered and canon in valid_sectors]
    if hits:
        return max(hits, key=lambda h: len(h[0]))[1]
    for canon in valid_sectors:
        if str(canon).lower() in lowered:
            return canon
    return None


LIVE_WORDS = ("live", "real-time", "real time", "right now",
              "currently", "as of now", "this morning", "at the moment")


def parse_live(text):
    # detecting a request to use current market data instead of the dataset
    lowered = text.lower()
    if any(w in lowered for w in LIVE_WORDS):
        return True
    m = re.search(r"\b(20\d{2})-\d{2}-\d{2}\b", lowered)
    return bool(m and int(m.group(1)) >= 2017)


def parse_horizon(text):
    # reading a multi-day horizon like 'next 5 days' or 'next week'
    lowered = text.lower()
    m = re.search(r"next\s+(\d{1,2})\s+(?:trading\s+)?days?", lowered)
    if m:
        return int(m.group(1))
    if "next week" in lowered:
        return 5
    if "next month" in lowered:
        return 21
    return None


def detect_intent(text, ticker, sector):
    # classifying the question so the chatbot can route it properly
    lowered = text.lower()
    if any(w in lowered for w in ("confidence", "how sure", "how certain",
                                  "certainty")):
        return "confidence"
    if any(w in lowered for w in ("why", "causing", "features", "reasons",
                                  "driving", "explain", "because of what")):
        return "explain"
    if re.search(r"(which|what|top|best)\s+(\d+\s+)?stocks", lowered) or \
            ("highest" in lowered and "stock" in lowered):
        return "top_movers"
    if re.search(r"(which|what)\s+sector", lowered):
        return "sector_rank"
    if sector is not None and ticker is None:
        return "sector_check"
    if "should i" in lowered or re.search(r"\b(sell or hold|buy or sell|"
                                          r"hold or sell)\b", lowered):
        return "advise"
    return "predict"


def parse_unknown_ticker(text):
    # accepting any plausible uppercase symbol when live mode is requested
    stopwords = {"I", "A", "UP", "DOWN", "ON", "IN", "AT", "TO", "IS", "IT",
                 "GO", "DO", "BE", "OR", "SO", "USE", "THE", "CNN", "RF",
                 "LIVE"}
    for token in re.findall(r"\b[A-Z][A-Z0-9.\-]{1,5}\b", text):
        if token not in stopwords:
            return token
    return None


def parse_query(text, tickers, name_map, latest_date, default_model,
                valid_sectors=None):
    # combining all extractors into one structured query dict
    ticker = parse_ticker(text, tickers, name_map)
    sector = parse_sector(text, valid_sectors or set())
    live = parse_live(text)
    if live and ticker is None:
        ticker = parse_unknown_ticker(text)
    return {"ticker": ticker,
            "sector": sector,
            "live": live,
            "date": parse_date(text, latest_date),
            "model": parse_model(text) or default_model,
            "horizon_days": parse_horizon(text),
            "intent": detect_intent(text, ticker, sector)}
