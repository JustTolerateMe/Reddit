"""Text analysis utilities for sentiment and topic tagging."""

from __future__ import annotations

from dataclasses import dataclass

from textblob import TextBlob


@dataclass(slots=True)
class AnalyzedPost:
    """Minimal analyzed record for downstream aggregation."""

    id: str
    title: str
    body: str
    sentiment_polarity: float
    sentiment_label: str
    topics: list[str]


def analyze_post(post: dict, keywords: list[str]) -> AnalyzedPost:
    """Run sentiment analysis and keyword-based topic extraction on one post."""
    title = post.get("title", "")
    body = post.get("selftext", "")
    text = f"{title}\n{body}".strip()

    polarity = TextBlob(text).sentiment.polarity if text else 0.0
    label = _sentiment_label(polarity)
    topics = tag_topics(text, keywords)

    return AnalyzedPost(
        id=post.get("id", ""),
        title=title,
        body=body,
        sentiment_polarity=polarity,
        sentiment_label=label,
        topics=topics,
    )


def tag_topics(text: str, keywords: list[str]) -> list[str]:
    """Simple keyword-based topic tagging."""
    text_lower = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in text_lower]


def _sentiment_label(polarity: float) -> str:
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return "neutral"
