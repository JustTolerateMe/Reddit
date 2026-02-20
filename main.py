from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from aggregate import brand_mentions, bubble_chart_data, topic_summary
from analyze import DATE_CUTOFF, analyze_comments, analyze_posts

TOPIC_KEYWORDS: Dict[str, Iterable[str]] = {
    "Pricing": ["price", "pricing", "cost", "expensive", "cheap", "value"],
    "Reliability": ["reliable", "downtime", "bug", "broken", "issue", "stable"],
    "Support": ["support", "help", "ticket", "response", "service"],
    "Features": ["feature", "roadmap", "integration", "api", "ui", "ux"],
    "Security": ["security", "privacy", "breach", "compliance", "gdpr", "soc2"],
}

TOPIC_CATEGORY_MAP: Dict[str, str] = {
    "Pricing": "Business",
    "Reliability": "Product",
    "Support": "Operations",
    "Features": "Product",
    "Security": "Trust",
}

BRAND_KEYWORDS: Dict[str, Iterable[str]] = {
    "OpenAI": ["openai", "chatgpt", "gpt"],
    "Google": ["google", "gemini", "bard"],
    "Anthropic": ["anthropic", "claude"],
    "Meta": ["meta", "llama"],
    "Microsoft": ["microsoft", "copilot", "azure"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Reddit posts/comments and export analytics tables.")
    parser.add_argument("--posts", required=True, help="Path to input posts CSV")
    parser.add_argument("--comments", required=True, help="Path to input comments CSV")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--min-score", type=int, default=3, help="Minimum score filter for analyzed output")
    parser.add_argument("--chunk-size", type=int, default=50, help="Incremental persistence chunk size")
    return parser.parse_args()


def _incremental_analyze(
    df: pd.DataFrame,
    analyzer,
    output_path: Path,
    chunk_size: int,
    **analyzer_kwargs,
) -> pd.DataFrame:
    if output_path.exists():
        output_path.unlink()

    analyzed_parts = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        analyzed_chunk = analyzer(chunk, **analyzer_kwargs)
        analyzed_parts.append(analyzed_chunk)
        analyzed_chunk.to_csv(output_path, mode="a", index=False, header=not output_path.exists())
    if not analyzed_parts:
        return analyzer(df, **analyzer_kwargs)
    return pd.concat(analyzed_parts, ignore_index=True)


def _summary_stats(posts_df: pd.DataFrame, comments_df: pd.DataFrame, topic_df: pd.DataFrame) -> Dict[str, object]:
    timestamps = pd.concat(
        [posts_df.get("created_ts", pd.Series(dtype="datetime64[ns, UTC]")), comments_df.get("created_ts", pd.Series(dtype="datetime64[ns, UTC]"))],
        ignore_index=True,
    ).dropna()

    min_date = timestamps.min().isoformat() if not timestamps.empty else None
    max_date = timestamps.max().isoformat() if not timestamps.empty else None

    anxiety = "N/A"
    joy = "N/A"
    if not topic_df.empty:
        anxiety = topic_df.sort_values("Avg_Sentiment", ascending=True).iloc[0]["Topic"]
        joy = topic_df.sort_values("Avg_Sentiment", ascending=False).iloc[0]["Topic"]

    return {
        "totals": {
            "raw_posts": int(len(posts_df)),
            "raw_comments": int(len(comments_df)),
            "analyzed_posts": int(posts_df["passes_filters"].sum()) if "passes_filters" in posts_df else 0,
            "analyzed_comments": int(comments_df["passes_filters"].sum()) if "passes_filters" in comments_df else 0,
        },
        "date_range": {"min": min_date, "max": max_date, "cutoff": DATE_CUTOFF.isoformat()},
        "top_topics": {"anxiety": anxiety, "joy": joy},
        "flags": {
            "posts_empty_deleted": int(posts_df.get("flag_empty_deleted", pd.Series(dtype=bool)).sum()),
            "posts_zero_comments": int(posts_df.get("flag_zero_comments", pd.Series(dtype=bool)).sum()),
            "posts_duplicate_ids": int(posts_df.get("flag_duplicate_id", pd.Series(dtype=bool)).sum()),
            "comments_empty_deleted": int(comments_df.get("flag_empty_deleted", pd.Series(dtype=bool)).sum()),
            "comments_duplicate_ids": int(comments_df.get("flag_duplicate_id", pd.Series(dtype=bool)).sum()),
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    posts_raw = pd.read_csv(args.posts)
    comments_raw = pd.read_csv(args.comments)

    posts_raw.to_csv(output_dir / "raw_posts.csv", index=False)
    comments_raw.to_csv(output_dir / "raw_comments.csv", index=False)

    posts_analyzed = _incremental_analyze(
        posts_raw,
        analyze_posts,
        output_dir / "posts_analyzed.csv",
        chunk_size=args.chunk_size,
        topic_keywords=TOPIC_KEYWORDS,
        topic_category_map=TOPIC_CATEGORY_MAP,
        min_score=args.min_score,
    )
    comments_analyzed = _incremental_analyze(
        comments_raw,
        analyze_comments,
        output_dir / "comments_analyzed.csv",
        chunk_size=args.chunk_size,
        topic_keywords=TOPIC_KEYWORDS,
        topic_category_map=TOPIC_CATEGORY_MAP,
        min_score=args.min_score,
    )

    posts_filtered = posts_analyzed[posts_analyzed["passes_filters"]].copy()
    comments_filtered = comments_analyzed[comments_analyzed["passes_filters"]].copy()

    topic_df = topic_summary(posts_filtered, comments_filtered)
    brand_df = brand_mentions(posts_filtered, comments_filtered, BRAND_KEYWORDS)
    bubble_df = bubble_chart_data(topic_df)

    topic_df.to_csv(output_dir / "topic_summary.csv", index=False)
    brand_df.to_csv(output_dir / "brand_mentions.csv", index=False)
    bubble_df.to_csv(output_dir / "bubble_chart_data.csv", index=False)

    scrape_log = _summary_stats(posts_analyzed, comments_analyzed, topic_df)
    (output_dir / "scrape_log.json").write_text(json.dumps(scrape_log, indent=2), encoding="utf-8")

    print("\n=== Processing Summary ===")
    print(f"Raw posts: {scrape_log['totals']['raw_posts']}")
    print(f"Raw comments: {scrape_log['totals']['raw_comments']}")
    print(f"Analyzed posts: {scrape_log['totals']['analyzed_posts']}")
    print(f"Analyzed comments: {scrape_log['totals']['analyzed_comments']}")
    print(
        f"Date range: {scrape_log['date_range']['min']} to {scrape_log['date_range']['max']} "
        f"(cutoff {scrape_log['date_range']['cutoff']})"
    )
    print(f"Top anxiety topic: {scrape_log['top_topics']['anxiety']}")
    print(f"Top joy topic: {scrape_log['top_topics']['joy']}")


if __name__ == "__main__":
    main()
