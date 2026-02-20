from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from textblob import TextBlob

DATE_CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
DELETED_MARKERS = {"", "[deleted]", "[removed]", "deleted", "removed", "none", "null"}


@dataclass(frozen=True)
class SentimentResult:
    polarity: float
    subjectivity: float
    sentiment_class: str


def sentiment_bucket(polarity: float) -> str:
    if polarity <= -0.6:
        return "Very Negative"
    if polarity <= -0.15:
        return "Negative"
    if polarity < 0.15:
        return "Neutral"
    if polarity < 0.6:
        return "Positive"
    return "Very Positive"


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def sentiment_for_text(text: str) -> SentimentResult:
    if not text.strip():
        return SentimentResult(0.0, 0.0, "Neutral")
    sentiment = TextBlob(text).sentiment
    polarity = float(sentiment.polarity)
    subjectivity = float(sentiment.subjectivity)
    return SentimentResult(polarity, subjectivity, sentiment_bucket(polarity))


def detect_topics(text: str, topic_keywords: Dict[str, Iterable[str]]) -> List[str]:
    lowered = text.lower()
    found: List[str] = []
    for topic, keywords in topic_keywords.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            found.append(topic)
    return found


def map_category(topics: List[str], topic_category_map: Dict[str, str]) -> str:
    for topic in topics:
        if topic in topic_category_map:
            return topic_category_map[topic]
    return "Other"


def is_empty_deleted(text: str) -> bool:
    return text.strip().lower() in DELETED_MARKERS


def _to_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    elif pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed


def analyze_posts(
    posts_df: pd.DataFrame,
    topic_keywords: Dict[str, Iterable[str]],
    topic_category_map: Dict[str, str],
    min_score: int = 3,
    date_cutoff: pd.Timestamp = DATE_CUTOFF,
) -> pd.DataFrame:
    df = posts_df.copy()
    df["title"] = df.get("title", "").map(_normalize_text)
    df["selftext"] = df.get("selftext", "").map(_normalize_text)
    df["text"] = (df["title"] + " " + df["selftext"]).str.strip()

    sentiments = df["text"].map(sentiment_for_text)
    df["polarity"] = sentiments.map(lambda s: s.polarity)
    df["subjectivity"] = sentiments.map(lambda s: s.subjectivity)
    df["sentiment_class"] = sentiments.map(lambda s: s.sentiment_class)

    df["topics"] = df["text"].map(lambda t: detect_topics(t, topic_keywords))
    df["topic"] = df["topics"].map(lambda t: t[0] if t else "Uncategorized")
    df["category"] = df["topics"].map(lambda t: map_category(t, topic_category_map))

    df["created_ts"] = _to_timestamp(df["created_utc"])
    df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0).astype(int)
    df["num_comments"] = pd.to_numeric(df.get("num_comments", 0), errors="coerce").fillna(0).astype(int)

    df["flag_empty_deleted"] = df["text"].map(is_empty_deleted)
    df["flag_zero_comments"] = df["num_comments"] <= 0
    df["flag_duplicate_id"] = df.duplicated(subset=["id"], keep=False) if "id" in df.columns else False
    df["flag_before_cutoff"] = df["created_ts"] < date_cutoff
    df["flag_below_min_score"] = df["score"] < min_score

    df["passes_filters"] = ~(
        df["flag_empty_deleted"]
        | df["flag_before_cutoff"]
        | df["flag_below_min_score"]
        | df["flag_duplicate_id"]
    )
    df["topics"] = df["topics"].map(lambda topics: "; ".join(topics))
    return df


def analyze_comments(
    comments_df: pd.DataFrame,
    topic_keywords: Dict[str, Iterable[str]],
    topic_category_map: Dict[str, str],
    min_score: int = 3,
    date_cutoff: pd.Timestamp = DATE_CUTOFF,
) -> pd.DataFrame:
    df = comments_df.copy()
    df["body"] = df.get("body", "").map(_normalize_text)

    sentiments = df["body"].map(sentiment_for_text)
    df["polarity"] = sentiments.map(lambda s: s.polarity)
    df["subjectivity"] = sentiments.map(lambda s: s.subjectivity)
    df["sentiment_class"] = sentiments.map(lambda s: s.sentiment_class)

    df["topics"] = df["body"].map(lambda t: detect_topics(t, topic_keywords))
    df["topic"] = df["topics"].map(lambda t: t[0] if t else "Uncategorized")
    df["category"] = df["topics"].map(lambda t: map_category(t, topic_category_map))

    df["created_ts"] = _to_timestamp(df["created_utc"])
    df["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0).astype(int)

    df["flag_empty_deleted"] = df["body"].map(is_empty_deleted)
    df["flag_duplicate_id"] = df.duplicated(subset=["id"], keep=False) if "id" in df.columns else False
    df["flag_before_cutoff"] = df["created_ts"] < date_cutoff
    df["flag_below_min_score"] = df["score"] < min_score

    df["passes_filters"] = ~(
        df["flag_empty_deleted"]
        | df["flag_before_cutoff"]
        | df["flag_below_min_score"]
        | df["flag_duplicate_id"]
    )
    df["topics"] = df["topics"].map(lambda topics: "; ".join(topics))
    return df
