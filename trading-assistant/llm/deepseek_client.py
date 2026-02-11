import os
import json
import hashlib
import time
import random
import re
import requests
import trafilatura
from datetime import datetime
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article, Config
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CACHE_DIR = "cache"
CONTENT_CACHE_FILE = os.path.join(CACHE_DIR, "article_content_cache.json")
SUMMARY_CACHE_FILE = os.path.join(CACHE_DIR, "llm_summary_cache.json")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

os.makedirs(CACHE_DIR, exist_ok=True)

def _get_cache(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_cache(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f)

def _get_content_cache():
    return _get_cache(CONTENT_CACHE_FILE)

def _save_content_cache(data):
    _save_cache(CONTENT_CACHE_FILE, data)

def _get_summary_cache():
    return _get_cache(SUMMARY_CACHE_FILE)

def _save_summary_cache(data):
    _save_cache(SUMMARY_CACHE_FILE, data)

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

def _is_google_domain(url):
    """Checks if the domain is a Google property to be ignored."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        google_domains = [
            "news.google.com", 
            "google.com", 
            "www.google.com",
            "googleusercontent.com", 
            "lh3.googleusercontent.com", 
            "gstatic.com"
        ]
        return any(gd in domain for gd in google_domains)
    except:
        return False

def resolve_google_news_rss_url(url):
    """
    Resolves the real publisher URL from a Google News RSS link.
    Returns: (publisher_url, reason)
    """
    if not url:
        return None, "empty_url"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    }

    try:
        resp = http.get(url, headers=headers, allow_redirects=True, timeout=10)
        
        if not _is_google_domain(resp.url):
            return resp.url, "redirect_success"

        content = resp.text
        soup = BeautifulSoup(content, 'html.parser')

        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            candidate = canonical["href"]
            if candidate.startswith("http") and not _is_google_domain(candidate):
                return candidate, "canonical"

        og_url = soup.find("meta", property="og:url")
        if og_url and og_url.get("content"):
            candidate = og_url["content"]
            if candidate.startswith("http") and not _is_google_domain(candidate):
                return candidate, "og_url"

        meta_refresh = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
        if meta_refresh:
            content_attr = meta_refresh.get("content", "")
            match = re.search(r'url=([^;]+)', content_attr, re.I)
            if match:
                candidate = match.group(1).strip()
                if candidate.startswith("http") and not _is_google_domain(candidate):
                    return candidate, "meta_refresh"

        link_candidates = soup.find_all("a", href=True)
        for link in link_candidates:
            href = link['href']
            
            def is_valid_candidate(u):
                return u.startswith("http") and not _is_google_domain(u)

            if is_valid_candidate(href):
                return href, "outbound_link"
            
            if "url=" in href or "&url=" in href:
                try:
                    match_url = re.search(r'[?&]url=([^&]+)', href)
                    if match_url:
                        decoded = requests.utils.unquote(match_url.group(1))
                        if is_valid_candidate(decoded):
                            return decoded, "decoded_url_param"
                except:
                    pass

        return None, "no_publisher_link_found"

    except Exception as e:
        return None, f"request_error: {str(e)}"

def _fetch_article_text(url):
    """
    1. Resolves Google URL to Publisher URL.
    2. Extracts Text using Trafilatura -> Newspaper3k -> BS4.
    Returns: (text, status, resolved_url, debug_info)
    """
    debug_info = {
        "original_url": url,
        "resolved_url": None,
        "resolve_reason": None,
        "final_url": None,
        "method": None,
        "extracted_length": 0,
        "error": None
    }

    resolved_url, reason = resolve_google_news_rss_url(url)
    
    debug_info["resolved_url"] = resolved_url
    debug_info["resolve_reason"] = reason
    
    if not resolved_url:
        return None, "resolve_failed", None, debug_info

    debug_info["final_url"] = resolved_url

    try:
        t_downloaded = trafilatura.fetch_url(resolved_url)
        if t_downloaded:
            text = trafilatura.extract(t_downloaded, include_comments=False, favor_precision=True)
            if text and len(text) >= 800:
                debug_info["method"] = "trafilatura"
                debug_info["extracted_length"] = len(text)
                return text, "full_text", resolved_url, debug_info
        
        try:
            config = Config()
            config.browser_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            config.request_timeout = 10
            
            article = Article(resolved_url, config=config)
            article.download()
            article.parse()
            if article.text and len(article.text) >= 800:
                debug_info["method"] = "newspaper3k"
                debug_info["extracted_length"] = len(article.text)
                return article.text, "full_text", resolved_url, debug_info
        except Exception as e_np:
            debug_info["newspaper_error"] = str(e_np)

        headers = {
             "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = http.get(resolved_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg", "form"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)
            
            if len(clean_text) >= 800:
                debug_info["method"] = "bs4_fallback"
                debug_info["extracted_length"] = len(clean_text)
                return clean_text, "full_text", resolved_url, debug_info

        return None, "empty_extract", resolved_url, debug_info

    except Exception as e:
        debug_info["error"] = str(e)
        return None, "extraction_exception", resolved_url, debug_info

def _call_deepseek(messages, max_tokens=2000):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "Error: DEEPSEEK_API_KEY not found in environment variables."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "deepseek-chat", 
        "messages": messages,
        "temperature": 0.3, 
        "stream": False,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling DeepSeek API: {str(e)}"

def summarize_articles_hybrid(articles):
    """
    1. Resolve & Extract Text for each article.
    2. Summarize each FULL TEXT article individually to 1 paragraph.
    3. Feed Summaries + Snippets to Final Combine Step.
    """
    if not articles:
        return {
            "summary_text": "No articles to summarize.", 
            "stats": {"total": 0, "full_text": 0, "snippet_only": 0, "details": {}},
            "debug_data": []
        }

    articles_sig = sorted([f"{a['title']}|{a['link']}" for a in articles])
    articles_hash = hashlib.md5(json.dumps(articles_sig).encode()).hexdigest()
    
    summary_cache = _get_summary_cache()
    if articles_hash in summary_cache and "debug_data" in summary_cache[articles_hash]:
        return summary_cache[articles_hash]

    content_cache = _get_content_cache()
    
    individual_summaries = [] 
    
    stats = {
        "total": len(articles), 
        "full_text": 0, 
        "snippet_only": 0,
        "details": {}
    }
    
    debug_data_list = []
    content_cache_updated = False

    for article in articles:
        url = article.get("link")
        title = article.get("title", "")
        published = article.get("published", "")
        snippet = article.get("summary", "")
        
        final_text_content = None
        source_status = "snippet_only"
        fail_reason = None
        current_debug = {}
        
        if url in content_cache:
            c = content_cache[url]
            final_text_content = c.get("text")
            source_status = c.get("source_type", "snippet_only")
            fail_reason = c.get("fail_reason")
            current_debug = c.get("debug_info", {})
        else:
            if url:
                extracted, status, final_url, dbg = _fetch_article_text(url)
                current_debug = dbg
                
                if extracted:
                    final_text_content = extracted
                    source_status = "full_text"
                    fail_reason = None
                else:
                    final_text_content = None
                    source_status = "snippet_only"
                    fail_reason = status
                
                content_cache[url] = {
                    "text": final_text_content,
                    "source_type": source_status,
                    "fail_reason": fail_reason,
                    "final_url": final_url,
                    "debug_info": current_debug,
                    "fetched_at": time.time()
                }
                content_cache_updated = True
            else:
                fail_reason = "missing_url"

        summ_content = ""
        
        if source_status == "full_text":
            stats["full_text"] += 1
            
            cached_summ = content_cache[url].get("individual_summary")
            
            if cached_summ:
                summ_content = cached_summ
            else:
                prompt = f"""Summarize this news article in 3-4 bullet points focusing on financial facts.
Title: {title}
Text:
{final_text_content[:6000]}
"""
                summ_content = _call_deepseek([{"role": "user", "content": prompt}], max_tokens=300)
                
                content_cache[url]["individual_summary"] = summ_content
                content_cache_updated = True
                
            individual_summaries.append(f"Source: {title} ({published})\nSummary:\n{summ_content}")
            
        else:
            stats["snippet_only"] += 1
            if fail_reason:
                stats["details"][fail_reason] = stats["details"].get(fail_reason, 0) + 1
            
            summ_content = snippet
            individual_summaries.append(f"Source: {title} ({published})\nSnippet:\n{summ_content}")

        debug_data_list.append({
            "Resolved URL": current_debug.get("resolved_url") or "N/A",
            "resolve_reason": current_debug.get("resolve_reason") or "N/A",
            "Status": source_status, 
            "Method": current_debug.get("method", "N/A"),
            "Size": current_debug.get("extracted_length", 0)
        })

    if content_cache_updated:
        _save_content_cache(content_cache)

    combined_input = "\n\n".join(individual_summaries)
    
    final_prompt = f"""
You are a senior financial analyst.
Based on the following article summaries and snippets:

{combined_input}

Produce a consolidated report:
1. **Executive Summary**: 5-7 bullet points.
2. **Key Themes**: Group related news.
3. **Risks & Red Flags**: Negative indicators.
4. **Catalysts**: Forward-looking events.

Style: Professional, dense, fact-based.
"""
    final_summary_text = _call_deepseek([{"role": "user", "content": final_prompt}], max_tokens=1000)

    result = {
        "summary_text": final_summary_text,
        "stats": stats,
        "debug_data": debug_data_list
    }
    
    summary_cache[articles_hash] = result
    _save_summary_cache(summary_cache)
    
    return result
