"""

Chrome Version: 142.0.7444.176

"""

import time
import random
import json
import re
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------
# CONFIG — MUST UPDATE THESE
# -----------------------------

# 1️ Path to your ChromeDriver (matching Chrome 142)
#CHROMEDRIVER_PATH = r"C:\tools\chromedriver\chromedriver.exe"
CHROMEDRIVER_PATH = r"E://Qode_interview//chromedriver-win64//chromedriver-win64//chromedriver.exe"

# 2️ Path to your real Chrome User Data folder
CHROME_PROFILE_PATH = r"C://Users//nehag//AppData//Local//Google//Chrome//User Data//Profile 4"

# 3️ Your Chrome profile directory (Default / Profile 1 / Profile 2)
PROFILE_DIRECTORY = "Default"

TARGET_TWEETS_PER_TAG = 600
HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty"]

RAW_OUT = "output/raw_tweets.jsonl"
PARQUET_OUT = "output/data.parquet"


# Logging
logger = logging.getLogger("qode")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("qode.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(fh)


# -----------------------------
# Selenium Setup with Logged-in Profile
# -----------------------------
def get_driver():
    options = Options()
    options.add_argument(f"--user-data-dir={CHROME_PROFILE_PATH}")
    options.add_argument(f"--profile-directory={PROFILE_DIRECTORY}")

    # IMPORTANT: no headless mode (Google blocks headless login)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-features=RendererCodeIntegrity")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    return webdriver.Chrome(
        service=Service(CHROMEDRIVER_PATH),
        options=options
    )


# -----------------------------
# Tweet Parsing Logic
# -----------------------------
def parse_tweet(elem):
    """Extract tweet details from Twitter/X logged-in UI."""
    try:
        content = elem.find_element(By.XPATH, ".//div[@data-testid='tweetText']").text
    except:
        content = ""

    try:
        username = elem.find_element(By.XPATH, ".//span[contains(text(),'@')]").text
    except:
        username = ""

    try:
        timestamp = elem.find_element(By.TAG_NAME, "time").get_attribute("datetime")
    except:
        timestamp = ""

    hashtags = re.findall(r"#\w+", content)
    mentions = re.findall(r"@\w+", content)

    return {
        "username": username,
        "content": content,
        "timestamp": timestamp,
        "hashtags": hashtags,
        "mentions": mentions,
        "scraped_at": datetime.now(timezone.utc).isoformat()
    }


# -----------------------------
# Scraper for One Hashtag
# -----------------------------
def scrape_hashtag(tag, limit):
    driver = get_driver()
    encoded = tag.replace("#", "%23")

    url = f"https://twitter.com/search?q={encoded}&f=live"
    driver.get(url)
    time.sleep(3)

    tweets = []
    seen = set()
    scrolls = 0

    while len(tweets) < limit and scrolls < 2000:
        articles = driver.find_elements(By.XPATH, "//article")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(parse_tweet, a) for a in articles]
            for f in as_completed(futures):
                t = f.result()
                if not t or not t["content"]:
                    continue
                key = (t["username"], t["content"], t["timestamp"])
                if key not in seen:
                    seen.add(key)
                    tweets.append(t)
                    if len(tweets) >= limit:
                        break

        # scroll down
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)

        time.sleep(random.uniform(0.7, 1.3))
        scrolls += 1

    driver.quit()
    return tweets[:limit]


# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(t):
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# -----------------------------
# Sentiment / TF-IDF / Signal
# -----------------------------
def get_sentiment(texts):
    return [TextBlob(t).sentiment.polarity for t in texts]


def get_tfidf_features(texts):
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=50)
    reduced = svd.fit_transform(X)
    return reduced


def composite_signal(sent, tfidf):
    s = (sent - np.mean(sent)) / (np.std(sent) + 1e-9)
    t = (tfidf.mean(axis=1) - tfidf.mean()) / (tfidf.std() + 1e-9)
    return 0.5 * s + 0.5 * t


# -----------------------------
# Save Helpers
# -----------------------------
def save_raw(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_parquet(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path)


# -----------------------------
# MAIN
# -----------------------------
def main():
    all_data = []

    for tag in HASHTAGS:
        logger.info(f"Scraping {tag} ...")
        tweets = scrape_hashtag(tag, TARGET_TWEETS_PER_TAG)
        all_data.extend(tweets)
        logger.info(f"Collected {len(tweets)} tweets for {tag}")

    save_raw(all_data, RAW_OUT)

    df = pd.DataFrame(all_data)
    df["clean_text"] = df["content"].apply(clean_text)

    df["sentiment"] = get_sentiment(df["clean_text"])
    tfidf = get_tfidf_features(df["clean_text"])
    df["signal"] = composite_signal(df["sentiment"].values, tfidf)

    save_parquet(df, PARQUET_OUT)

    print("DONE.")
    print("Raw:", RAW_OUT)
    print("Parquet:", PARQUET_OUT)


if __name__ == "__main__":
    main()
