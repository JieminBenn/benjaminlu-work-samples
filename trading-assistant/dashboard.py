import streamlit as st
import pandas as pd
from data.sec import get_financial_metrics
from data.news import fetch_recent_news, analyze_news
from data.price import fetch_current_price
from analysis.metrics import compute_derived_metrics, assess_fundamentals
from analysis.report import generate_markdown_report
from llm.deepseek_client import summarize_articles_hybrid
import os
from dotenv import load_dotenv

load_dotenv() 


st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("News + Fundamentals Research Assistant")
st.markdown("Automated analysis of SEC filings and recent news.")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    days_back = st.slider("News Lookback (Days)", 7, 90, 14)

    run_btn = st.button("Run Research")

if "research_data" not in st.session_state:
    st.session_state.research_data = None

if run_btn and ticker:
    with st.spinner(f"Analyzing {ticker}..."):
        base_metrics = get_financial_metrics(ticker)
        
        if not base_metrics:
            st.error(f"Could not fetch financial data for {ticker}. Please check the ticker.")
            st.session_state.research_data = None
        else:
            news_articles = fetch_recent_news(ticker, days_back=days_back)
            news_analysis = analyze_news(news_articles)
            
            price = fetch_current_price(ticker)
            
            st.session_state.research_data = {
                "ticker": ticker,
                "base_metrics": base_metrics,
                "news_articles": news_articles,
                "news_analysis": news_analysis,
                "price": price
            }

if st.session_state.research_data:
    data = st.session_state.research_data
    ticker = data["ticker"]
    base_metrics = data["base_metrics"]
    news_articles = data["news_articles"]
    price = data["price"]
    
    valuation_display = {}
    if price:
        valuation_display["Price"] = price
        
        eps_data = base_metrics.get("eps_diluted_ttm")
        if eps_data and eps_data["value"] and eps_data["value"] != 0:
            pe = price / eps_data["value"]
            valuation_display["P/E (TTM)"] = f"{pe:.2f}"
        else:
            valuation_display["P/E (TTM)"] = "N/A"
    
    ni_data = base_metrics.get("net_income_ttm")
    eps_data = base_metrics.get("eps_diluted_ttm")
    shares_est = 0
    if ni_data and eps_data and eps_data["value"]:
        shares_est = ni_data["value"] / eps_data["value"]
    
    if price and shares_est:
        mkt_cap = price * shares_est
        valuation_display["Market Cap"] = f"${mkt_cap / 1e9:.2f}B"
        
        fcf_data = base_metrics.get("fcf_ttm")
        if fcf_data:
            yld = (fcf_data["value"] / mkt_cap) * 100
            valuation_display["FCF Yield (TTM)"] = f"{yld:.2f}%"
    
    st.subheader(f"Fundamentals (TTM) - Ended {base_metrics.get('fiscal_period_end', 'N/A')}")
    col1, col2, col3, col4 = st.columns(4)
    
    def fmt_money(val_obj):
        if not val_obj: return "N/A"
        val = val_obj["value"]
        return f"${val/1e9:.2f}B"
    
    with col1:
        st.metric("Revenue (TTM)", fmt_money(base_metrics.get("revenue_ttm")))
    with col2:
        st.metric("Net Income (TTM)", fmt_money(base_metrics.get("net_income_ttm")))
    with col3:
        eps = base_metrics.get("eps_diluted_ttm")
        st.metric("EPS Diluted (TTM)", f"${eps['value']:.2f}" if eps else "N/A")
    with col4:
        st.metric("Free Cash Flow (TTM)", fmt_money(base_metrics.get("fcf_ttm")))
        

    rev_fy = base_metrics.get("revenue_fy")
    fy_end = rev_fy["date"] if rev_fy else "N/A"
    
    st.subheader(f"Fundamentals (FY - Latest 10-K) - Ended {fy_end}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Revenue (FY)", fmt_money(base_metrics.get("revenue_fy")))
    with col2:
        st.metric("Net Income (FY)", fmt_money(base_metrics.get("net_income_fy")))
    with col3:
        eps_fy = base_metrics.get("eps_diluted_fy")
        st.metric("EPS Diluted (FY)", f"${eps_fy['value']:.2f}" if eps_fy else "N/A")
    with col4:
        st.metric("Free Cash Flow (FY)", fmt_money(base_metrics.get("fcf_fy")))
    
    rev_hist = base_metrics.get("revenue_history")
    if rev_hist:
        with st.expander("View Historical Annual Revenue"):
            hist_rows = []
            for h in rev_hist:
                hist_rows.append({
                    "Fiscal Year End": h["date"],
                    "Revenue": fmt_money(h)
                })
            st.table(pd.DataFrame(hist_rows))

    st.subheader("Balance Sheet (Latest Snapshot)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cash & Equiv", fmt_money(base_metrics.get("cash_snapshot")))
    with col2:
        st.metric("Total Debt", fmt_money(base_metrics.get("debt_snapshot")))
    with col3:
        de = base_metrics.get("debt_to_equity")
        st.metric("Debt/Equity", f"{de:.2f}" if de is not None else "N/A")
        
    st.subheader("Valuation")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Price", f"${price:.2f}" if price else "N/A")
    with col2:
        st.metric("Market Cap", valuation_display.get("Market Cap", "N/A"))
    with col3:
        st.metric("P/E (TTM)", valuation_display.get("P/E (TTM)", "N/A"))
    with col4:
        st.metric("FCF Yield (TTM)", valuation_display.get("FCF Yield (TTM)", "N/A"))
    
    st.divider()
    
    with st.expander("View Raw Metrics Details"):
        raw_rows = []
        
        METRIC_CONFIG = {
            "revenue_ttm": ("Revenue", "TTM"),
            "net_income_ttm": ("Net Income", "TTM"),
            "eps_diluted_ttm": ("EPS (Diluted)", "TTM"),
            "ocf_ttm": ("Operating Cash Flow", "TTM"),
            "capex_ttm": ("CapEx", "TTM"),
            "fcf_ttm": ("Free Cash Flow", "TTM"),
            
            "revenue_fy": ("Revenue", "FY (Latest 10-K)"),
            "net_income_fy": ("Net Income", "FY (Latest 10-K)"),
            "eps_diluted_fy": ("EPS (Diluted)", "FY (Latest 10-K)"),
            "ocf_fy": ("Operating Cash Flow", "FY (Latest 10-K)"),
            "capex_fy": ("CapEx", "FY (Latest 10-K)"),
            "fcf_fy": ("Free Cash Flow", "FY (Latest 10-K)"),
            
            "cash_snapshot": ("Cash & Equivalents", "Snapshot"),
            "debt_snapshot": ("Total Debt", "Snapshot"),
            "equity_snapshot": ("Stockholders' Equity", "Snapshot"),
            "debt_to_equity": ("Debt/Equity Ratio", "Snapshot"),
        }

        def format_val(val, key):
            if val is None: return "N/A"
            if isinstance(val, (int, float)):
                if "debt_to_equity" in key:
                    return f"{val:.2f}"
                if "yield" in key: 
                    return f"{val:.2f}%"
                if "eps" in key:
                    return f"${val:.2f}"
                
                abs_val = abs(val)
                if abs_val >= 1e9:
                    return f"${val/1e9:.2f}B"
                elif abs_val >= 1e6:
                    return f"${val/1e6:.2f}M"
                elif abs_val > 0:
                    return f"${val:,.2f}"
                else:
                    return "0"
                    
            return str(val)

        for k, v in base_metrics.items():
            
            val_to_show = None
            date_to_show = None
            form_to_show = None
            
            if isinstance(v, dict) and 'value' in v:
                val_to_show = v['value']
                date_to_show = v.get('date')
                form_to_show = v.get('form')
            elif k == "debt_to_equity" and isinstance(v, (int, float)):
                    val_to_show = v
                    date_to_show = "Latest Snapshot"
                    form_to_show = "Derived"
            
            if val_to_show is not None:
                label, period = METRIC_CONFIG.get(k, (k.replace("_", " ").title(), "Other"))
                
                if period == "Other":
                    if "_ttm" in k: period = "TTM"
                    elif "_fy" in k: period = "FY (Latest 10-K)"
                    elif "_snapshot" in k: period = "Snapshot"
                    
                fmt_val = format_val(val_to_show, k)
                
                if form_to_show:
                    if "Sum of" in form_to_show:
                        forms = form_to_show.replace("Sum of ", "").split(",")
                        if len(forms) == 4:
                            form_to_show = "TTM (4x 10-Q)"
                        else:
                            form_to_show = f"Sum ({len(forms)} filings)"
                
                row = {
                    "Period": period,
                    "Metric": label,
                    "Display Value": fmt_val,
                    "Period End": date_to_show,
                    "Source Form": form_to_show,
                    "Raw Key": k, 
                    "Raw Value": val_to_show 
                }
                raw_rows.append(row)

        if raw_rows:
            df = pd.DataFrame(raw_rows)
            
            period_order = {"TTM": 0, "FY (Latest 10-K)": 1, "Snapshot": 2, "Other": 3}
            df["_sort"] = df["Period"].map(lambda x: period_order.get(x, 99))
            df = df.sort_values(by=["_sort", "Metric"])
            
            st.dataframe(
                df[["Period", "Metric", "Display Value", "Period End", "Source Form", "Raw Key"]],
                column_config={
                    "Period": st.column_config.TextColumn("Period", help="Timeframe of the metric"),
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Display Value": st.column_config.TextColumn("Value", width="small"), 
                    "Period End": st.column_config.DateColumn("Period End", format="YYYY-MM-DD"),
                    "Source Form": st.column_config.TextColumn("Source"),
                    "Raw Key": st.column_config.TextColumn("Raw Key", disabled=True),
                },
                hide_index=True,
                use_container_width=True
            )
            
    st.divider()
    st.subheader("Recent News Headlines")
    
    if news_articles:
        df_news = pd.DataFrame(news_articles)[["published", "source", "title", "link"]]
        
        df_news["published"] = pd.to_datetime(df_news["published"], errors="coerce")
        df_news = df_news.sort_values(by="published", ascending=False)
        
        st.dataframe(
            df_news,
            column_config={
                "published": st.column_config.DateColumn("Date", format="MMM DD, YYYY", width="medium"),
                "source": st.column_config.TextColumn("Source", width="medium"),
                "title": st.column_config.TextColumn("Headline", width="large"),
                "link": st.column_config.LinkColumn("Read", display_text="🔗 Read"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.divider()
        if st.button("Summarize Articles (DeepSeek Hybrid)"):
            if "DEEPSEEK_API_KEY" not in os.environ:
                st.error("DeepSeek API Key not found. Please set DEEPSEEK_API_KEY environment variable.")
                st.info("You can set this in your terminal: export DEEPSEEK_API_KEY='your_key'")
            else:
                with st.spinner("Reading full articles and summarizing..."):
                    summary_res = summarize_articles_hybrid(news_articles)
                    
                    if isinstance(summary_res["summary_text"], str) and summary_res["summary_text"].startswith("Error"):
                        st.error(summary_res["summary_text"])
                    else:
                        st.subheader("DeepSeek Hybrid Summary")
                        st.markdown(summary_res["summary_text"])
                        
                        stats = summary_res["stats"]
                        details_text = ""
                        if stats.get("details"):
                            details_list = [f"{k}: {v}" for k, v in stats["details"].items()]
                            details_text = f"  \n(Fail Reasons: {', '.join(details_list)})"

                        st.info(
                            f"**Transparency Stats:**  \n"
                            f"Articles Summarized: {stats['total']}  \n"
                            f"Full-Text Success: {stats['full_text']} / {stats['total']}  \n"
                            f"Snippet-Only Fallback: {stats['snippet_only']} / {stats['total']}"
                            f"{details_text}"
                        )
                        
                        if summary_res.get("debug_data"):
                            with st.expander("View Extraction Debug Log"):
                                st.dataframe(pd.DataFrame(summary_res["debug_data"]))
    else:
        st.write("No recent news found.")

if not ticker:
    st.info("Enter a ticker symbol to begin.")
