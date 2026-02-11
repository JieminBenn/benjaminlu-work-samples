import yfinance as yf

def fetch_current_price(ticker):
    """
    Fetches the current stock price using yfinance.
    Returns float or None.
    """
    try:
        data = yf.Ticker(ticker)
        if hasattr(data, 'fast_info'):
            price = data.fast_info.get('last_price')
            if price:
                return price
                
        hist = data.history(period="1d")
        if not hist.empty:
            return hist["Close"].iloc[-1]
            
        return None
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return None
