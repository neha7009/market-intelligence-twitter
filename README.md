# market-intelligence-twitter
Market intelligence &amp; signal extraction from Twitter data
# Market Intelligence from Twitter Data

This project scrapes Twitter/X market-related hashtags using a logged-in Chrome user profile, 
then analyzes sentiment signals to detect bullish/bearish market trends.

## How to Run

### 1️⃣ Install requirements
pip install -r requirements.txt

### 2️⃣ Run scraper
python main_market_24h_2000_scraper.py

### 3️⃣ Run analysis
python analyze_tweets.py

Outputs stored in `output/`

---

## Features
✔ Multi-hashtag scraping  
✔ Sentiment analysis  
✔ TF-IDF intelligence signals  
✔ Parquet + CSV outputs  
✔ Deduplication using hashing  
✔ Logged-in browser to bypass anti-bot  

