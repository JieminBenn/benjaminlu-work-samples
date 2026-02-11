
import argparse
import sys
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()


from data.sec import get_financial_metrics
from data.news import fetch_recent_news, analyze_news
from data.price import fetch_current_price
from analysis.metrics import compute_derived_metrics, assess_fundamentals
from analysis.report import generate_markdown_report

def main():
    parser = argparse.ArgumentParser(description="Research Assistant: News + Fundamentals")
    parser.add_argument("ticker", help="Company ticker symbol (e.g. AAPL)")
    parser.add_argument("--days", type=int, default=14, help="Days back for news search (default: 14)")
    parser.add_argument("--price", type=float, help="Current share price (optional, will fetch if missing)")
    parser.add_argument("--json", help="Output to JSON file (optional)")
    
    args = parser.parse_args()
    
    console = Console()
    
    with console.status(f"[bold green]Fetching data for {args.ticker}...[/bold green]"):
        current_price = args.price
        if not current_price:
            console.log(f"Fetching current price for {args.ticker}...")
            current_price = fetch_current_price(args.ticker)
            if current_price:
                console.log(f"Current price: ${current_price:.2f}")
            else:
                console.log("[yellow]Could not fetch current price. Valuation metrics will be skipped.[/yellow]")

        base_metrics = get_financial_metrics(args.ticker)
        if not base_metrics:
            console.print(f"[bold red]Error:[/bold red] Could not fetch financial data for {args.ticker}. Check ticker or connection.")
            sys.exit(1)
            
        news_articles = fetch_recent_news(args.ticker, days_back=args.days)
        news_analysis = analyze_news(news_articles)
        
        metrics = compute_derived_metrics(base_metrics, price=current_price)
        fundamentals = assess_fundamentals(metrics)
        
        report_md = generate_markdown_report(args.ticker, metrics, fundamentals, news_analysis, news_articles)
        
    console.print(Markdown(report_md))
    
    if args.json:
        import json
        output_data = {
            "metrics": metrics,
            "fundamentals": fundamentals,
            "news_analysis": news_analysis,
            "news_articles": news_articles
        }
        with open(args.json, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        console.print(f"\n[green]Report saved to {args.json}[/green]")

if __name__ == "__main__":
    main()
