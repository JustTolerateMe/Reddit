"""Aggregation helpers for reporting and visualization payloads."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import pandas as pd

from .analyze import AnalyzedPost


def to_frame(analyzed_posts: list[AnalyzedPost]) -> pd.DataFrame:
    """Convert analyzed dataclasses to a DataFrame."""
    return pd.DataFrame(asdict(post) for post in analyzed_posts)


def topic_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build counts of topic mentions."""
    if frame.empty:
        return pd.DataFrame(columns=["topic", "mentions"])

    exploded = frame.explode("topics")
    summary = (
        exploded.dropna(subset=["topics"])
        .groupby("topics", as_index=False)
        .size()
        .rename(columns={"topics": "topic", "size": "mentions"})
        .sort_values("mentions", ascending=False)
    )
    return summary


def brand_mentions(frame: pd.DataFrame, brands: list[str]) -> dict[str, int]:
    """Count brand mentions in title/body text."""
    counts = Counter()
    for _, row in frame.iterrows():
        text = f"{row.get('title', '')} {row.get('body', '')}".lower()
        for brand in brands:
            if brand.lower() in text:
                counts[brand] += 1
    return dict(counts)


def bubble_chart_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a chart-friendly dataset with topic, volume, and average sentiment."""
    if frame.empty:
        return pd.DataFrame(columns=["topic", "volume", "avg_sentiment"])

    exploded = frame.explode("topics").dropna(subset=["topics"])
    return (
        exploded.groupby("topics", as_index=False)
        .agg(volume=("id", "count"), avg_sentiment=("sentiment_polarity", "mean"))
        .rename(columns={"topics": "topic"})
        .sort_values("volume", ascending=False)
    )
