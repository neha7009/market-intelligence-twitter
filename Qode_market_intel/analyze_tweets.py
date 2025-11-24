"""

"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


DATA_PARQUET = "output/data.parquet"
SIGNALS_CSV = "output/signals.csv"
SUMMARY_CSV = "output/hashtag_summary.csv"


# ---------- helpers ----------

def clean_text(t: str) -> str:
    import re
    if not isinstance(t, str):
        return ""
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "clean_text" not in df.columns:
        df["clean_text"] = df["content"].astype(str).apply(clean_text)

    df["sentiment"] = df["clean_text"].apply(
        lambda txt: TextBlob(txt).sentiment.polarity if txt else 0.0
    )
    return df


def add_tfidf_signal(df: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
    texts = df["clean_text"].fillna("").tolist()

    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(X)          # shape: (n_samples, n_components)

    # simple 1-D TF–IDF strength: mean over components
    tfidf_strength = reduced.mean(axis=1)

    # z-score sentiment & tfidf_strength, then average -> composite signal
    sent = df["sentiment"].values
    s_z = (sent - sent.mean()) / (sent.std() + 1e-9)
    t_z = (tfidf_strength - tfidf_strength.mean()) / (tfidf_strength.std() + 1e-9)

    df["tfidf_strength"] = tfidf_strength
    df["signal"] = 0.5 * s_z + 0.5 * t_z

    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    # original timestamp column from your scraper
    ts_col = "timestamp" if "timestamp" in df.columns else "scraped_at"

    df["timestamp_utc"] = pd.to_datetime(df[ts_col], errors="coerce")
    # if you want IST:
    df["timestamp_ist"] = df["timestamp_utc"].dt.tz_convert("Asia/Kolkata")

    # hourly bucket for aggregation
    df["hour_bucket"] = df["timestamp_utc"].dt.floor("H")
    return df


# ---------- main analysis ----------

def main():
    path = Path(DATA_PARQUET)
    if not path.exists():
        raise FileNotFoundError(
            f"{DATA_PARQUET} not found. Run the scraper first to generate it."
        )

    print(f"Reading {DATA_PARQUET} ...")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} tweets")

    # 1) basic cleaning + sentiment
    df = add_sentiment(df)

    # 2) tf-idf + composite signal
    df = add_tfidf_signal(df, n_components=50)

    # 3) time buckets for intraday behaviour
    df = add_time_columns(df)

    # 4) save full per-tweet signals dataset (for modelling)
    Path(SIGNALS_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SIGNALS_CSV, index=False)
    print(f"Saved per-tweet signals to {SIGNALS_CSV}")

    # 5) aggregate summary per hashtag and hour
    group_cols = ["hour_bucket", "hashtags"]
    # if your `hashtags` column is list-like, explode it:
    if df["hashtags"].dtype == "object":
        df_expl = df.copy()
        df_expl = df_expl.explode("hashtags")
        df_expl["hashtags"] = df_expl["hashtags"].fillna("")
    else:
        df_expl = df

    summary = (
        df_expl.groupby(["hour_bucket", "hashtags"])
        .agg(
            tweet_count=("content", "count"),
            avg_sentiment=("sentiment", "mean"),
            avg_signal=("signal", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved hashtag/hour summary to {SUMMARY_CSV}")

    # 6) simple example plots (for notebook-style analysis)

    # sentiment over time for NIFTY50 tweets
    mask_nifty = df_expl["hashtags"].str.contains("nifty50", case=False, na=False)
    nifty_ts = (
        df_expl[mask_nifty]
        .groupby("hour_bucket")["sentiment"]
        .mean()
        .sort_index()
    )

    plt.figure()
    nifty_ts.plot()
    plt.title("Average Sentiment over Time – #nifty50")
    plt.xlabel("Time (hour)")
    plt.ylabel("Sentiment")
    plt.tight_layout()
    plt.savefig("output/nifty50_sentiment_timeseries.png")
    print("Saved plot: output/nifty50_sentiment_timeseries.png")

    # bar chart of avg signal by main hashtag
    main_tags = ["#nifty50", "#sensex", "#banknifty", "#intraday"]
    avg_by_tag = []
    for tag in main_tags:
        m = df_expl["hashtags"].str.contains(tag.replace("#", ""), case=False, na=False)
        if m.any():
            avg_by_tag.append((tag, df_expl.loc[m, "signal"].mean()))

    if avg_by_tag:
        tags, vals = zip(*avg_by_tag)
        plt.figure()
        plt.bar(tags, vals)
        plt.title("Average Composite Signal by Hashtag")
        plt.ylabel("Signal")
        plt.tight_layout()
        plt.savefig("output/hashtag_signal_bar.png")
        print("Saved plot: output/hashtag_signal_bar.png")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
