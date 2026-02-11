
from datetime import datetime

def determine_horizon(quality, risk, news_analysis):
    """
    Determines the "Horizon Fit" based on inputs.
    """
    themes = news_analysis.get_themes() if hasattr(news_analysis, 'get_themes') else news_analysis.get("themes", [])
    red_flags = news_analysis.get("red_flags", [])
    
    fit = "Unclear/Avoid"
    rationale = []
    
    catalysts = [t for t in themes if t in ["Earnings", "M&A", "Product", "Regulation"]]
    if catalysts and "High" not in risk:
        fit = "Short-term catalyst watch"
        rationale.append(f"Active catalysts: {', '.join(catalysts)}")
    elif red_flags:
        fit = "Avoid / High Risk"
        rationale.append(f"Red flags present: {', '.join(red_flags)}")
        return fit, rationale
        
    if quality == "Strong" and risk in ["Low", "Medium"]:
        fit = "Long-term compounder candidate"
        rationale.append("Strong fundamentals + manageable risk profile")
    
    if fit == "Unclear/Avoid" and news_analysis.get("sentiment_label") == "Positive" and risk != "High":
        fit = "Medium-term setup"
        rationale.append("Positive sentiment momentum with acceptable risk")
        
    if not rationale:
        rationale.append("Insufficient data to form a strong view.")
        
    return fit, rationale

def generate_markdown_report(ticker, metrics, fundamental_analysis, news_analysis, news_articles):
    """
    Generates the full markdown report string.
    """
    
    m = metrics or {}
    f = fundamental_analysis or {}
    n = news_analysis or {}
    
    horizon, horizon_rationale = determine_horizon(f.get("quality"), f.get("risk"), n)
    
    def fmt(val, fmt_str="${:,.2f}"):
        if val is None: return "N/A"
        try:
            return fmt_str.format(val)
        except:
            return str(val)
            
    def fmt_b(val):
        if val is None: return "N/A"
        return f"${val / 1e9:.2f}B"
        
    def fmt_pct(val):
        if val is None: return "N/A"
        return f"{val*100:.1f}%"

    price_str = f"${m.get('price'):.2f}" if m.get('price') else "Not provided"
    
    report = []
    report.append(f"# Research Report: {ticker.upper()}")
    report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Sources**: SEC Filings, Google News")
    report.append("\n---\n")
    
    report.append("## Research Conclusion (News + Fundamentals Only)")
    report.append("> **Disclaimer**: For research/education; not investment advice.\n")
    
    report.append(f"- **Fundamental Quality**: {f.get('quality')} ({', '.join(f.get('rationale', []))})")
    report.append(f"- **Financial Risk**: {f.get('risk')}")
    report.append(f"- **News Regime**: {n.get('sentiment_label')} (Score: {n.get('sentiment_score')})")
    report.append(f"- **Horizon Fit**: **{horizon}**")
    for r in horizon_rationale:
        report.append(f"  - {r}")
        
    report.append("\n---\n")
    
    report.append("## Key Fundamentals")
    
    report.append("| Metric | Value | Reference |")
    report.append("|---|---|---|")
    report.append(f"| Revenue (TTM) | {fmt_b(m.get('revenue'))} | {m.get('fiscal_period_end')} |")
    report.append(f"| Net Income | {fmt_b(m.get('net_income'))} | |")
    report.append(f"| Net Margin | {fmt_pct(m.get('net_margin'))} | |")
    report.append(f"| EPS (Diluted) | {fmt(m.get('eps_diluted'))} | |")
    report.append(f"| Free Cash Flow | {fmt_b(m.get('free_cash_flow'))} | |")
    
    report.append("\n**Balance Sheet**")
    report.append("| Metric | Value |")
    report.append("|---|---|")
    report.append(f"| Cash & Eq. | {fmt_b(m.get('cash_equivalents'))} |")
    report.append(f"| Total Debt | {fmt_b(m.get('total_debt'))} |")
    report.append(f"| Dept/Equity | {fmt(m.get('debt_to_equity'), '{:.2f}')} |")
    
    if m.get('price'):
        report.append("\n**Valuation**")
        report.append(f"- Price: {price_str}")
        report.append(f"- Market Cap: {fmt_b(m.get('market_cap'))}")
        report.append(f"- P/E Ratio: {fmt(m.get('pe_ratio'), '{:.1f}')}")
    else:
        report.append("\n*(Valuation metrics skipped: No price provided)*")
        
    report.append("\n---\n")
    
    report.append(f"## Recent News Analysis (Last {len(news_articles)} articles analyzed)")
    
    if n.get("red_flags"):
        report.append(f"> [!WARNING] **Red Flags Detected**: {', '.join(n.get('red_flags'))}")
        
    if n.get("themes"):
        report.append(f"- **Top Themes**: {', '.join(n.get('themes'))}")
        
    report.append("\n### Top Headlines")
    report.append("| Date | Source | Headline |")
    report.append("|---|---|---|")
    
    for art in news_articles[:10]:
        title = (art['title'][:75] + '..') if len(art['title']) > 75 else art['title']
        date_str = str(art['published'])[:16] 
        report.append(f"| {date_str} | {art['source']} | [{title}]({art['link']}) |")
        
    return "\n".join(report)
