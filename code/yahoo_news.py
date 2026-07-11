"""fetching recent Yahoo Finance headlines for a ticker via yfinance

kept intentionally lightweight: a short, sentiment-free digest of recent
company headlines that the chatbot can attach to an answer, with graceful
fallback when yfinance's news field is empty or the network is down
"""

MAX_HEADLINES = 5


def fetch_news(ticker):
    # returning a list of recent {title, publisher, link, age} dicts
    try:
        import time
        import yfinance as yf
        t = yf.Ticker(ticker.replace(".", "-"))
        raw = getattr(t, "news", None) or []
    except Exception:
        return []

    items = []
    now = time.time()
    for a in raw[:MAX_HEADLINES * 2]:
        # yfinance news shape has shifted over versions; read defensively
        content = a.get("content", a) if isinstance(a, dict) else {}
        title = (content.get("title") or a.get("title") or "").strip()
        if not title:
            continue
        pub = (content.get("provider", {}) or {}).get("displayName") \
            or a.get("publisher") or "Yahoo Finance"
        link = ""
        if isinstance(content.get("canonicalUrl"), dict):
            link = content["canonicalUrl"].get("url", "")
        link = link or a.get("link", "")
        ts = a.get("providerPublishTime")
        age = ""
        if ts:
            hrs = max(0, (now - ts) / 3600)
            age = f"{int(hrs)}h ago" if hrs < 48 else f"{int(hrs / 24)}d ago"
        items.append({"title": title, "publisher": pub,
                      "link": link, "age": age})
        if len(items) >= MAX_HEADLINES:
            break
    return items


def news_block(ticker):
    # formatting a short plain-text headline digest, or a clear empty note
    items = fetch_news(ticker)
    if not items:
        return (f"No recent Yahoo Finance headlines came back for "
                f"{ticker.upper()} (the feed is sometimes empty or "
                f"unavailable).")
    lines = [f"Recent Yahoo Finance headlines for {ticker.upper()}:"]
    for it in items:
        age = f" · {it['age']}" if it["age"] else ""
        lines.append(f"  - {it['title']} ({it['publisher']}{age})")
    lines.append("\nHeadlines are context only — the model does not read "
                 "them, and news is typically priced within minutes.")
    return "\n".join(lines)
