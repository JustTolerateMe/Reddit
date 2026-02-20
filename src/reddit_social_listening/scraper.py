"""Reddit JSON scraping helpers with pagination and retry/backoff."""

from __future__ import annotations

import time
from typing import Any

import requests

from .config import PipelineConfig

USER_AGENT = "reddit-social-listening/0.1"


def fetch_subreddit_posts(
    subreddit: str,
    config: PipelineConfig,
    *,
    listing: str = "new",
    limit: int = 100,
    max_pages: int = 5,
) -> list[dict[str, Any]]:
    """Fetch posts for a subreddit using Reddit's public JSON endpoint."""
    posts: list[dict[str, Any]] = []
    after: str | None = None

    for _ in range(max_pages):
        payload = _request_listing(subreddit, config, listing=listing, limit=limit, after=after)
        children = payload.get("data", {}).get("children", [])
        if not children:
            break

        posts.extend(item.get("data", {}) for item in children)
        after = payload.get("data", {}).get("after")
        if not after:
            break

        _respect_rate_limit(config)

    return posts


def _request_listing(
    subreddit: str,
    config: PipelineConfig,
    *,
    listing: str,
    limit: int,
    after: str | None,
) -> dict[str, Any]:
    """Execute a single request with retries and exponential backoff."""
    url = f"https://www.reddit.com/r/{subreddit}/{listing}.json"
    params = {"limit": limit}
    if after:
        params["after"] = after

    for attempt in range(config.max_retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=config.request_timeout_s,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt >= config.max_retries:
                raise
            delay = config.backoff_base_s**attempt
            time.sleep(delay)

    return {}


def _respect_rate_limit(config: PipelineConfig) -> None:
    """Sleep according to configured request rate."""
    if config.requests_per_second <= 0:
        return
    time.sleep(1 / config.requests_per_second)
