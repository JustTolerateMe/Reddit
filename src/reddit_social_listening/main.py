"""CLI orchestration for scraping, analyzing, and exporting Reddit insights."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from .aggregate import bubble_chart_data, to_frame, topic_summary
from .analyze import analyze_post
from .config import DEFAULT_CONFIG, PipelineConfig
from .io_utils import ensure_output_dir, incremental_jsonl_write, save_csv, save_json, write_log
from .scraper import fetch_subreddit_posts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reddit social listening starter CLI")
    parser.add_argument("--output-dir", default="output", help="Directory for generated files")
    parser.add_argument("--max-pages", type=int, default=3, help="Pages per subreddit to fetch")
    return parser


def run_pipeline(config: PipelineConfig, output_dir: str, max_pages: int) -> None:
    output = ensure_output_dir(output_dir)
    all_posts: list[dict] = []

    for subreddit in config.subreddits:
        posts = fetch_subreddit_posts(subreddit, config, max_pages=max_pages)
        all_posts.extend(posts)
        write_log(f"Fetched {len(posts)} posts from r/{subreddit}", output / "pipeline.log")

    analyzed = [analyze_post(post, config.keywords) for post in tqdm(all_posts, desc="Analyzing posts")]
    frame = to_frame(analyzed)

    save_csv(frame, output / "posts_analyzed.csv")
    save_csv(topic_summary(frame), output / "topic_summary.csv")
    save_csv(bubble_chart_data(frame), output / "bubble_chart_data.csv")
    incremental_jsonl_write([asdict(item) for item in analyzed], output / "posts_analyzed.jsonl")
    save_json(
        {
            "total_posts": len(all_posts),
            "subreddits": config.subreddits,
            "keywords": config.keywords,
            "output_files": [
                "posts_analyzed.csv",
                "topic_summary.csv",
                "bubble_chart_data.csv",
                "posts_analyzed.jsonl",
            ],
        },
        output / "run_metadata.json",
    )


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(DEFAULT_CONFIG, output_dir=args.output_dir, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
