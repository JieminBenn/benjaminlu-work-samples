import feedparser
import os
import hashlib
import json
import time
from datetime import datetime, timedelta
from textblob import TextBlob

CACHE_DIR = "cache"
RSS_URL_TEMPLATE = "https://news.google.com/rss/search?q={ticker}+when:{days}d&hl=en-US&gl=US&ceid=US:en"

os.makedirs(CACHE_DIR, exist_ok=True)

def _get_cache_path(ticker, days_back):
    return os.path.join(CACHE_DIR, f"news_{ticker}_{days_back}.json")

def _load_cache(ticker, days_back):
    path = _get_cache_path(ticker, days_back)
    if os.path.exists(path):
        if time.time() - os.path.getmtime(path) < 1800: 
            with open(path, "r") as f:
                return json.load(f)
    return None

def _save_cache(ticker, days_back, data):
    with open(_get_cache_path(ticker, days_back), "w") as f:
        json.dump(data, f)

def fetch_recent_news(ticker, days_back=14):
    """
    Fetches news from Google News RSS.
    Returns a list of dicts: {title, link, published, summary, source}
    """
    cached = _load_cache(ticker, days_back)
    if cached:
        return cached

    url = RSS_URL_TEMPLATE.format(ticker=ticker, days=days_back)
    feed = feedparser.parse(url)
    
    articles = []
    seen_titles = set()
    
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    for entry in feed.entries[:30]: 
        title = entry.title
        if title in seen_titles:
            continue
        seen_titles.add(title)
        
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                 published_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                 if published_dt < cutoff_date:
                     continue
        except Exception:
            pass 
            
        articles.append({
            "title": title,
            "link": entry.link,
            "published": entry.published,
            "summary": entry.summary if 'summary' in entry else "",
            "source": entry.source.title if 'source' in entry else "Unknown"
        })
        
    _save_cache(ticker, days_back, articles)
    return articles

def analyze_news(articles):
    """
    Analyzes a list of articles for sentiment and red flags.
    Returns a dict with analysis results.
    """
    if not articles:
        return {"sentiment_score": 0, "sentiment_label": "Neutral", "red_flags": [], "themes": []}
        
    polarity_sum = 0
    red_flag_keywords = [
        "lawsuit", "sued", "fraud", "investigation", "SEC", "DOJ", "bankruptcy", 
        "downgrade", "recall", "breach", "restatement", "accounting error", "scandal"
    ]
    
    themes_map = {
        "Earnings": ["earnings", "report", "revenue", "eps", "profit", "quarter"],
        "Product": ["launch", "release", "device", "new feature", "upgrade"],
        "Regulation": ["regulation", "antitrust", "ban", "compliance"],
        "M&A": ["acquisition", "merger", "buyout", "takeover"],
        "Legal": ["court", "trial", "settlement", "judge", "legal"],
        "Macro": ["inflation", "rate", "fed", "economy", "recession"]
    }
    
    found_red_flags = set()
    found_themes = set()
    
    for article in articles:
        text = f"{article['title']} {article['summary']}"
        blob = TextBlob(text)
        polarity_sum += blob.sentiment.polarity
        
        lower_text = text.lower()
        
        for rf in red_flag_keywords:
            if rf in lower_text:
                found_red_flags.add(rf)
                
        for theme, keywords in themes_map.items():
            if any(k in lower_text for k in keywords):
                found_themes.add(theme)
                
    avg_polarity = polarity_sum / len(articles)
    
    label = "Neutral"
    if avg_polarity > 0.1:
        label = "Positive"
    elif avg_polarity < -0.1:
        label = "Negative"
        
    return {
        "sentiment_score": round(avg_polarity, 2),
        "sentiment_label": label,
        "red_flags": list(found_red_flags),
        "themes": list(found_themes)
    }
