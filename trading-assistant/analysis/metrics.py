
def compute_derived_metrics(metrics, price=None):
    """
    Augments the metrics dictionary with derived ratios and margins.
    metrics: dict from sec.get_financial_metrics
    price: float or None
    """
    if not metrics:
        return {}
    
    m = metrics.copy()
    
    rev = m.get("revenue")
    op_inc = m.get("operating_income")
    net_inc = m.get("net_income")
    
    if rev and rev > 0:
        m["operating_margin"] = (op_inc / rev) if op_inc is not None else None
        m["net_margin"] = (net_inc / rev) if net_inc is not None else None
    else:
        m["operating_margin"] = None
        m["net_margin"] = None

    curr_assets = m.get("total_assets") 
    curr_liab = m.get("total_liabilities") 
    
    total_debt = m.get("total_debt", 0)
    equity = m.get("shareholder_equity")
    if equity and equity > 0:
        m["debt_to_equity"] = total_debt / equity
    else:
        m["debt_to_equity"] = None 
        
    m["price"] = price
    if price:
        shares = None
        if net_inc and m.get("eps_basic"):
            shares = net_inc / m["eps_basic"]
        
        m["market_cap"] = (shares * price) if shares else None
        
        if m.get("eps_basic") and m["eps_basic"] > 0:
            m["pe_ratio"] = price / m["eps_basic"]
        else:
            m["pe_ratio"] = None
            
        if rev and m.get("market_cap"):
            m["ps_ratio"] = m["market_cap"] / rev 
        else:
            m["ps_ratio"] = None
            
    else:
        m["market_cap"] = None
        m["pe_ratio"] = None
        m["ps_ratio"] = None
        
    return m

def assess_fundamentals(metrics):
    """
    Returns a dict with 'quality_label' and 'risk_label' and 'rationale'.
    """
    if not metrics:
        return {"quality": "Unknown", "risk": "Unknown", "rationale": ["No data available."]}
        
    reasons = []
    
    score = 0
    
    if metrics.get("net_margin") and metrics["net_margin"] > 0.15:
        score += 1
        reasons.append("High net margin (>15%)")
    elif metrics.get("net_margin") and metrics["net_margin"] > 0.05:
        score += 0.5
    elif metrics.get("net_margin") and metrics["net_margin"] < 0:
        score -= 1
        reasons.append("Negative net margin")
        
    if metrics.get("free_cash_flow") and metrics["free_cash_flow"] > 0:
        score += 1
        reasons.append("Positive Free Cash Flow")
    else:
        score -= 1
        reasons.append("Negative or missing Free Cash Flow")
        
    quality = "Mixed"
    if score >= 1.5:
        quality = "Strong"
    elif score <= -1:
        quality = "Weak"
        
    risk_score = 0
    de = metrics.get("debt_to_equity")
    if de is not None:
        if de > 2.0:
            risk_score += 1
            reasons.append(f"High Debt/Equity ({de:.2f})")
        elif de < 0.5:
            reasons.append("Conservative leverage")
            
    cash = metrics.get("cash_equivalents", 0) or 0
    debt = metrics.get("total_debt", 0) or 0
    if cash > debt:
        risk_score -= 1
        reasons.append("Net cash position (Cash > Debt)")
    elif debt > (cash * 3):
        risk_score += 1
        reasons.append("High debt relative to cash")

    risk = "Medium"
    if risk_score >= 1:
        risk = "High"
    elif risk_score <= -1:
        risk = "Low"
        
    return {
        "quality": quality,
        "risk": risk,
        "rationale": reasons
    }
