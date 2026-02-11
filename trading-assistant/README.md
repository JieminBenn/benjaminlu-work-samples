
# Automated Financial Research Assistant

## Project Overview

This application is a **News + Fundamentals Research Assistant**. It automates the initial phase of stock research by aggregating quantitative hard data with qualitative sentiment analysis.

### Core Features
1.  **SEC Financials**: Directly queries the SEC EDGAR database to parse raw XBRL data, extracting key metrics like Revenue, Net Income, and Free Cash Flow without relying on third-party financial APIs.
2.  **News Sentiment Pipeline**: Scrapes Google News RSS feeds, resolves redirects to original publisher URLs, and extracts full article text for analysis.
3.  **Hybrid AI Summarization**: Uses a local keyword/sentiment scoring system (TextBlob) alongside an optional Large Language Model integration (DeepSeek) to generate executive summaries of recent news.
4.  **Interactive Dashboard**: A Streamlit-based frontend that allows users to visualize financial trends and explore news themes dynamically.

## Installation

### Prerequisites
- Python 3.8+
- A valid DeepSeek API Key (optional, for advanced summarization features)

### Setup
1.  Clone the repository and create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  (Optional) Set up environment variables for AI features:
    ```bash
    export DEEPSEEK_API_KEY="your_api_key_here"
    ```
    *Or create a `.env` file in the root directory.*

## Usage

### 1. Web Dashboard (Recommended)
Launch the interactive interface to explore data visually.
```bash
streamlit run dashboard.py
```
This will open the application in your default browser at `http://localhost:8501`.

### 2. Command Line Interface (CLI)
Run quick analysis directly from your terminal.
```bash
python app.py AAPL
```

**Options:**
- `--days N`: Look back N days for news (default: 14).
- `--price X.XX`: Manually provide current share price if the automatic fetch fails.
- `--json file.json`: Save the full analysis output to a JSON file.

Example:
```bash
python app.py NVDA --days 7 --json nvda_report.json
```

## Project Structure

- **`data/`**: Handles all external data fetching (SEC EDGAR, Google News, Yahoo Finance).
- **`analysis/`**: Contains the business logic for computing financial ratios and scoring sentiment.
- **`llm/`**: Manages interactions with the DeepSeek API for text summarization.
- **`app.py`**: CLI entry point.
- **`dashboard.py`**: Streamlit web application entry point.

---
*Disclaimer: This project is for educational and demonstration purposes only. It is not financial advice.*
