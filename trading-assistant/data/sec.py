import requests
import json
import os
import time
from datetime import datetime
import pandas as pd

SEC_USER_AGENT = "ResearchAssistant/1.0 (contact@example.com)" 
CACHE_DIR = "cache"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

os.makedirs(CACHE_DIR, exist_ok=True)

def _get_headers():
    return {"User-Agent": SEC_USER_AGENT}

def _load_cache(key):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        if time.time() - os.path.getmtime(path) < 1800:
            with open(path, "r") as f:
                return json.load(f)
    return None

def _save_cache(key, data):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f)

def fetch_company_tickers():
    """Fetches the SEC ticker mapping."""
    cache_key = "sec_tickers"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    try:
        resp = requests.get(SEC_TICKERS_URL, headers=_get_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        _save_cache(cache_key, data)
        return data
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return {}

def get_cik(ticker):
    """Converts ticker to CIK (string with 10 digits)."""
    tickers = fetch_company_tickers()
    ticker = ticker.upper()
    
    for _, info in tickers.items():
        if info["ticker"] == ticker:
            return f"{info['cik_str']:010d}"
    
    return None

def fetch_company_facts(cik):
    """Fetches XBRL company facts."""
    cache_key = f"facts_{cik}"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    url = SEC_FACTS_URL.format(cik=cik)
    try:
        resp = requests.get(url, headers=_get_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        _save_cache(cache_key, data)
        return data
    except Exception as e:
        print(f"Error fetching facts for {cik}: {e}")
        return None

def _parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return None

def _extract_snapshot(concept_data):
    """
    Extracts the latest snapshot value (BS items).
    """
    if not concept_data:
        return None

    valid_entries = []
    for x in concept_data:
        if 'val' not in x or 'end' not in x:
            continue
        valid_entries.append(x)
    
    if not valid_entries:
        return None

    valid_entries.sort(key=lambda x: (x.get('end', ''), x.get('filed', '')), reverse=True)
    
    best = valid_entries[0]
    return {
        "value": best['val'],
        "date": best['end'],
        "form": best.get('form', 'Unknown'),
        "filed": best.get('filed', 'Unknown')
    }

def _extract_ttm_sum(concept_data):
    """
    Calculates TTM by summing the most recent 4 quarters.
    """
    if not concept_data:
        return None

    quarterly_entries = []
    for x in concept_data:
        if 'val' not in x or 'start' not in x or 'end' not in x:
            continue
        
        start = _parse_date(x['start'])
        end = _parse_date(x['end'])
        if not start or not end:
            continue
        
        days = (end - start).days
        if 75 <= days <= 105:
            quarterly_entries.append(x)
            
    if not quarterly_entries:
        return None

    quarterly_entries.sort(key=lambda x: (x.get('end', ''), x.get('filed', '')), reverse=True)
    
    unique_quarters = []
    seen_ends = set()
    for q in quarterly_entries:
        if q['end'] not in seen_ends:
            unique_quarters.append(q)
            seen_ends.add(q['end'])
            
    if len(unique_quarters) < 4:
        return None

    top_4 = unique_quarters[:4]
    
    total_val = sum(q['val'] for q in top_4)
    latest_end = top_4[0]['end']
    forms = [q.get('form', '?') for q in top_4]
    
    return {
        "value": total_val,
        "date": latest_end, 
        "form": f"Sum of {','.join(forms)}",
        "filed": top_4[0].get('filed', 'Unknown'),
        "is_ttm": True
    }

def _extract_latest_fy(concept_data):
    """
    Extracts the latest Fiscal Year (FY) value from 10-K filings.
    """
    if not concept_data:
        return None

    entries = []
    for x in concept_data:
        if 'val' not in x or 'end' not in x:
            continue
        
        if x.get('fp') == 'FY' and '10-K' in x.get('form', ''):
            entries.append(x)
    
    if not entries:
        return None

    entries.sort(key=lambda x: (x.get('end', ''), x.get('filed', '')), reverse=True)
    
    best = entries[0]
    return {
        "value": best['val'],
        "date": best['end'],
        "form": best.get('form', 'Unknown'),
        "filed": best.get('filed', 'Unknown')
    }

def _extract_fy_history(concept_data, limit=5):
    """
    Extracts historical FY values from 10-K filings. Returns list of dicts.
    """
    if not concept_data:
        return []

    entries = []
    for x in concept_data:
        if 'val' not in x or 'end' not in x:
            continue
        if x.get('fp') == 'FY' and '10-K' in x.get('form', ''):
            entries.append(x)
            
    if not entries:
        return []
        
    entries.sort(key=lambda x: (x.get('end', ''), x.get('filed', '')), reverse=True)
    
    history = []
    seen_ends = set()
    for x in entries:
        if x['end'] not in seen_ends:
            history.append({
                "value": x['val'],
                "date": x['end'],
                "form": x.get('form', 'Unknown'),
                "filed": x.get('filed', 'Unknown')
            })
            seen_ends.add(x['end'])
            if len(history) >= limit:
                break
                
    return history
def get_financial_metrics(ticker):
    """
    Orchestrates fetching and parsing into a standardized dict.
    Returns None if ticker not found or error.
    """
    cik = get_cik(ticker)
    if not cik:
        print(f"Ticker {ticker} not found in SEC database.")
        return None
        
    facts = fetch_company_facts(cik)
    if not facts or "facts" not in facts or "us-gaap" not in facts["facts"]:
        return None
    
    gaap = facts["facts"]["us-gaap"]
    
    def get_tag_data(tag):
        if tag in gaap:
            units = gaap[tag]["units"]
            if "USD" in units: return units["USD"]
            if "shares" in units: return units["shares"]
            if units: return units[list(units.keys())[0]]
        return []

    def get_best_metric(tags, extractor_func):
        """
        Iterates through all tags, runs extractor, returns the result with the latest date.
        """
        best_result = None
        
        for tag in tags:
            data = get_tag_data(tag)
            res = extractor_func(data)
            if res:
                if best_result is None or res['date'] > best_result['date']:
                    best_result = res
        
        return best_result

    rev_tags = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]
    rev_data = get_best_metric(rev_tags, _extract_ttm_sum)
    
    ni_tags = ["NetIncomeLoss", "ProfitLoss"]
    ni_data = get_best_metric(ni_tags, _extract_ttm_sum)
    
    eps_tags = ["EarningsPerShareDiluted"]
    eps_data = get_best_metric(eps_tags, _extract_ttm_sum)
    
    ocf_tags = ["NetCashProvidedByUsedInOperatingActivities"]
    ocf_data = get_best_metric(ocf_tags, _extract_ttm_sum)
    
    capex_tags = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
    capex_data = get_best_metric(capex_tags, _extract_ttm_sum)
    
    rev_fy = get_best_metric(rev_tags, _extract_latest_fy)
    ni_fy = get_best_metric(ni_tags, _extract_latest_fy)
    eps_fy = get_best_metric(eps_tags, _extract_latest_fy)
    ocf_fy = get_best_metric(ocf_tags, _extract_latest_fy)
    capex_fy = get_best_metric(capex_tags, _extract_latest_fy)
    
    rev_history = []
    for tag in rev_tags:
        data = get_tag_data(tag)
        hist = _extract_fy_history(data, 5)
        if hist:
            if not rev_history or (hist[0]['date'] > rev_history[0]['date']):
                rev_history = hist
    
    cash_tags = ["CashAndCashEquivalentsAtCarryingValue", "Cash", "CashAndDueFromBanks"]
    cash_data = get_best_metric(cash_tags, _extract_snapshot)
    
    debt_current_tags = ["CommercialPaper", "DebtCurrent", "ShortTermDebt", "NotesPayableCurrent"]
    debt_noncurrent_tags = ["LongTermDebtNoncurrent", "LongTermDebt"]
    
    d_curr = get_best_metric(debt_current_tags, _extract_snapshot)
    d_long = get_best_metric(debt_noncurrent_tags, _extract_snapshot)
    
    total_debt_val = 0
    debt_date = "N/A"
    
    if d_curr:
        total_debt_val += d_curr["value"]
        debt_date = d_curr["date"]
    
    if d_long:
        total_debt_val += d_long["value"]
        if d_long["date"] > debt_date: 
            debt_date = d_long["date"]
            
    debt_data = {
        "value": total_debt_val,
        "date": debt_date,
        "form": "Calc",
        "filed": "N/A"
    } if (d_curr or d_long) else None

    equity_tags = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
    equity_data = get_best_metric(equity_tags, _extract_snapshot)

    metrics = {
        "ticker": ticker,
        "cik": cik,
        
        "revenue_ttm": rev_data,
        "net_income_ttm": ni_data,
        "eps_diluted_ttm": eps_data,
        "ocf_ttm": ocf_data,
        "capex_ttm": capex_data,
        
        "revenue_fy": rev_fy,
        "net_income_fy": ni_fy,
        "eps_diluted_fy": eps_fy,
        "ocf_fy": ocf_fy,
        "capex_fy": capex_fy,
        "revenue_history": rev_history,
        
        "cash_snapshot": cash_data,
        "debt_snapshot": debt_data,
        "equity_snapshot": equity_data,
        
        "fiscal_period_end": rev_data["date"] if rev_data else (cash_data["date"] if cash_data else "N/A"),
        "latest_filing_date": rev_data["filed"] if rev_data else "N/A"
    }
    
    if ocf_data and capex_data:
        metrics["fcf_ttm"] = {
            "value": ocf_data["value"] - capex_data["value"],
            "date": ocf_data["date"],
            "form": "Calc"
        }
    else:
        metrics["fcf_ttm"] = None

    if ocf_fy and capex_fy:
        metrics["fcf_fy"] = {
            "value": ocf_fy["value"] - capex_fy["value"],
            "date": ocf_fy["date"],
            "form": "Calc (FY)"
        }
    else:
        metrics["fcf_fy"] = None

    if debt_data and equity_data and equity_data["value"] != 0:
        metrics["debt_to_equity"] = debt_data["value"] / equity_data["value"]
    else:
        metrics["debt_to_equity"] = None
        
    metrics["revenue"] = rev_data["value"] if rev_data else None
    metrics["net_income"] = ni_data["value"] if ni_data else None
    
    return metrics
