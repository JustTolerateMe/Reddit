from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

NEGATIVE_CLASSES = {"Very Negative", "Negative"}
POSITIVE_CLASSES = {"Positive", "Very Positive"}


def _explode_topics(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    temp = df.copy()
    temp["Topic"] = temp["topics"].fillna("").map(
        lambda value: [topic.strip() for topic in str(value).split(";") if topic.strip()]
    )
    temp = temp.explode("Topic")
    temp = temp[temp["Topic"].notna() & (temp["Topic"] != "")]
    if text_col in temp.columns:
        temp["_sample_text"] = temp[text_col].astype(str)
    else:
        temp["_sample_text"] = ""
    return temp


def topic_summary(posts_analyzed: pd.DataFrame, comments_analyzed: pd.DataFrame) -> pd.DataFrame:
    posts_topics = _explode_topics(posts_analyzed, "text")
    comments_topics = _explode_topics(comments_analyzed, "body")
    combined = pd.concat([posts_topics, comments_topics], ignore_index=True, sort=False)

    if combined.empty:
        return pd.DataFrame(
            columns=[
                "Topic",
                "Category",
                "Avg_Sentiment",
                "Mention_Count",
                "Pct_Negative",
                "Pct_Positive",
                "Top_Subreddit",
                "Sample_Quote",
            ]
        )

    grouped = combined.groupby("Topic", dropna=False)
    summary = grouped.agg(
        Category=("category", lambda s: s.mode().iat[0] if not s.mode().empty else "Other"),
        Avg_Sentiment=("polarity", "mean"),
        Mention_Count=("Topic", "count"),
        Pct_Negative=("sentiment_class", lambda s: (s.isin(NEGATIVE_CLASSES).mean()) * 100.0),
        Pct_Positive=("sentiment_class", lambda s: (s.isin(POSITIVE_CLASSES).mean()) * 100.0),
        Top_Subreddit=("subreddit", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
        Sample_Quote=("_sample_text", lambda s: str(s.iloc[0])[:220]),
    ).reset_index()

    summary = summary.sort_values(["Mention_Count", "Avg_Sentiment"], ascending=[False, False])
    return summary


def brand_mentions(
    posts_analyzed: pd.DataFrame,
    comments_analyzed: pd.DataFrame,
    brand_keywords: Dict[str, Iterable[str]],
) -> pd.DataFrame:
    sources: List[pd.DataFrame] = []

    posts = posts_analyzed.copy()
    posts["context"] = posts.get("text", "")
    posts["post_url"] = posts.get("url", posts.get("permalink", ""))
    sources.append(posts)

    comments = comments_analyzed.copy()
    comments["context"] = comments.get("body", "")
    comments["post_url"] = comments.get("link_permalink", "")
    sources.append(comments)

    combined = pd.concat(sources, ignore_index=True, sort=False)

    rows = []
    for _, row in combined.iterrows():
        context = str(row.get("context", ""))
        lowered = context.lower()
        for brand, keywords in brand_keywords.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                rows.append(
                    {
                        "Brand": brand,
                        "Subreddit": row.get("subreddit", ""),
                        "Sentiment": row.get("sentiment_class", "Neutral"),
                        "Context_Snippet": context[:220],
                        "Post_URL": row.get("post_url", ""),
                    }
                )

    return pd.DataFrame(rows, columns=["Brand", "Subreddit", "Sentiment", "Context_Snippet", "Post_URL"])


def bubble_chart_data(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=["Topic", "Category", "X_Sentiment", "Y_Position", "Bubble_Size", "Color_Code"]
        )

    categories = {category: idx for idx, category in enumerate(sorted(summary_df["Category"].dropna().unique()), start=1)}
    data = summary_df.copy()
    data["X_Sentiment"] = data["Avg_Sentiment"]
    data["Y_Position"] = data["Category"].map(categories).fillna(0)
    data["Bubble_Size"] = data["Mention_Count"]
    data["Color_Code"] = data["Category"].map(lambda category: f"C{categories.get(category, 0)}")
    return data[["Topic", "Category", "X_Sentiment", "Y_Position", "Bubble_Size", "Color_Code"]]
