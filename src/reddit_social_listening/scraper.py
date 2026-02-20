"""Reddit scraping utilities for social listening pipelines.

This module provides resilient request helpers, subreddit keyword search,
post normalization, pagination, deduplication, and limited-depth comment
collection.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


DEFAULT_USER_AGENT = (
    "reddit-social-listening-bot/1.0 "
    "(public-json-client; +https://www.reddit.com/)"
)
MIN_REQUEST_DELAY_SECONDS = 6.0
MAX_RETRIES = 3
RATE_LIMIT_WAIT_SECONDS = 60
DEFAULT_TIMEOUT = 20
SCRAPE_LOG_PATH = Path("scrape_log.json")

_LAST_REQUEST_TS = 0.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_scrape_log(event_type: str, **payload: Any) -> None:
    """Append a structured event record to scrape_log.json."""
    entry = {
        "timestamp": _utc_now_iso(),
        "event_type": event_type,
        **payload,
    }
    with SCRAPE_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _emit_progress(message: str, **payload: Any) -> None:
    """Emit progress to stdout and write the corresponding structured log."""
    print(f"[scraper] {message}")
    _append_scrape_log("progress", message=message, **payload)


def _enforce_min_delay() -> None:
    global _LAST_REQUEST_TS
    now = time.time()
    elapsed = now - _LAST_REQUEST_TS
    if _LAST_REQUEST_TS and elapsed < MIN_REQUEST_DELAY_SECONDS:
        wait_for = MIN_REQUEST_DELAY_SECONDS - elapsed
        _emit_progress("Waiting to respect minimum request delay", seconds=wait_for)
        time.sleep(wait_for)
    _LAST_REQUEST_TS = time.time()


def request_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Any]:
    """Request JSON with retries, pacing, and resilient error handling.

    Behavior:
    - Applies a custom User-Agent by default.
    - Enforces >= 6 second delay between outbound requests.
    - Retries up to MAX_RETRIES with exponential backoff.
    - Waits explicit 60 seconds for HTTP 429 before retrying.
    - Handles HTTP 403/404 gracefully (logs and returns None).
    """
    merged_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged_headers.update(headers)

    for attempt in range(1, MAX_RETRIES + 1):
        _enforce_min_delay()
        try:
            response = requests.get(
                url,
                params=params,
                headers=merged_headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            _append_scrape_log(
                "error",
                stage="request_json",
                url=url,
                params=params,
                attempt=attempt,
                error=str(exc),
            )
            if attempt == MAX_RETRIES:
                return None
            wait_for = 2 ** (attempt - 1)
            _emit_progress(
                "Transient request error. Retrying with backoff.",
                url=url,
                attempt=attempt,
                wait_seconds=wait_for,
            )
            time.sleep(wait_for)
            continue

        status = response.status_code
        if status == 200:
            try:
                return response.json()
            except ValueError as exc:
                _append_scrape_log(
                    "error",
                    stage="request_json",
                    url=url,
                    params=params,
                    attempt=attempt,
                    status_code=status,
                    error=f"Invalid JSON payload: {exc}",
                )
                return None

        if status in (403, 404):
            _append_scrape_log(
                "warning",
                stage="request_json",
                url=url,
                params=params,
                status_code=status,
                reason="Graceful handling for forbidden/not-found",
            )
            return None

        if status == 429:
            _append_scrape_log(
                "warning",
                stage="request_json",
                url=url,
                params=params,
                attempt=attempt,
                status_code=status,
                wait_seconds=RATE_LIMIT_WAIT_SECONDS,
                reason="Rate limited by Reddit",
            )
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
            continue

        _append_scrape_log(
            "error",
            stage="request_json",
            url=url,
            params=params,
            attempt=attempt,
            status_code=status,
            response_text=response.text[:500],
        )
        if attempt == MAX_RETRIES:
            return None

        wait_for = 2 ** (attempt - 1)
        time.sleep(wait_for)

    return None


def normalize_post(
    raw_post: Dict[str, Any],
    search_keyword: str,
    time_filter: str,
) -> Dict[str, Any]:
    """Normalize a Reddit post object into the required downstream schema."""
    post_id = raw_post.get("id")
    return {
        "post_id": post_id,
        "subreddit": raw_post.get("subreddit"),
        "title": raw_post.get("title", ""),
        "selftext": raw_post.get("selftext", ""),
        "score": raw_post.get("score", 0),
        "num_comments": raw_post.get("num_comments", 0),
        "created_utc": raw_post.get("created_utc"),
        "url": raw_post.get("url"),
        "search_keyword": [search_keyword],
        "time_filter": [time_filter],
    }


def merge_post_metadata(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge metadata fields for duplicate post IDs while preserving all matches."""
    for field in ("search_keyword", "time_filter"):
        left = set(existing.get(field, []))
        right = set(incoming.get(field, []))
        existing[field] = sorted(left.union(right))
    return existing


def search_subreddit_keyword(
    subreddit: str,
    keyword: str,
    time_filter: str = "year",
    pages: int = 1,
    per_page_limit: int = 100,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Search subreddit posts by keyword with pagination and normalization.

    Uses Reddit public JSON search endpoint and supports the `t` filter values,
    including `year` and `month`.
    """
    if time_filter not in {"hour", "day", "week", "month", "year", "all"}:
        raise ValueError(f"Unsupported time_filter: {time_filter}")

    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    after: Optional[str] = None
    collected: List[Dict[str, Any]] = []

    _emit_progress(
        "Starting subreddit keyword search",
        subreddit=subreddit,
        keyword=keyword,
        time_filter=time_filter,
        pages=pages,
    )

    for page in range(1, max(1, pages) + 1):
        params = {
            "q": keyword,
            "restrict_sr": 1,
            "sort": "new",
            "t": time_filter,
            "limit": per_page_limit,
            "raw_json": 1,
        }
        if after:
            params["after"] = after

        payload = request_json(url, params=params, headers=headers)
        if not payload:
            _append_scrape_log(
                "warning",
                stage="search_subreddit_keyword",
                subreddit=subreddit,
                keyword=keyword,
                time_filter=time_filter,
                page=page,
                reason="No payload returned",
            )
            break

        data = payload.get("data", {})
        children = data.get("children", [])
        for child in children:
            post_data = child.get("data", {})
            normalized = normalize_post(post_data, keyword, time_filter)
            if normalized.get("post_id"):
                collected.append(normalized)

        after = data.get("after")
        _emit_progress(
            "Fetched search page",
            subreddit=subreddit,
            keyword=keyword,
            time_filter=time_filter,
            page=page,
            page_posts=len(children),
            next_after=after,
        )

        if not after:
            break

    return collected


def deduplicate_posts(posts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate posts by post_id while merging keyword/time-filter metadata."""
    posts_list = list(posts)
    by_id: Dict[str, Dict[str, Any]] = {}
    for post in posts_list:
        post_id = post.get("post_id")
        if not post_id:
            continue
        if post_id not in by_id:
            by_id[post_id] = post
        else:
            by_id[post_id] = merge_post_metadata(by_id[post_id], post)

    deduped = list(by_id.values())
    _emit_progress(
        "Deduplicated posts",
        input_count=len(posts_list),
        output_count=len(deduped),
    )
    return deduped


def fetch_post_comments(
    subreddit: str,
    post_id: str,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch top-level comments and first-level replies for a post."""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    params = {
        "limit": 100,
        "depth": 2,
        "sort": "top",
        "raw_json": 1,
    }
    payload = request_json(url, params=params, headers=headers)
    if not payload or not isinstance(payload, list) or len(payload) < 2:
        _append_scrape_log(
            "warning",
            stage="fetch_post_comments",
            subreddit=subreddit,
            post_id=post_id,
            reason="No comments payload returned",
        )
        return []

    comments_listing = payload[1].get("data", {}).get("children", [])
    extracted: List[Dict[str, Any]] = []

    for comment in comments_listing:
        comment_data = comment.get("data", {})
        if not comment_data.get("id"):
            continue

        top_level = {
            "comment_id": comment_data.get("id"),
            "parent_id": comment_data.get("parent_id"),
            "body": comment_data.get("body", ""),
            "score": comment_data.get("score", 0),
            "created_utc": comment_data.get("created_utc"),
            "depth": 0,
        }
        extracted.append(top_level)

        replies = comment_data.get("replies", {})
        if isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            for reply in reply_children:
                reply_data = reply.get("data", {})
                if not reply_data.get("id"):
                    continue
                extracted.append(
                    {
                        "comment_id": reply_data.get("id"),
                        "parent_id": reply_data.get("parent_id"),
                        "body": reply_data.get("body", ""),
                        "score": reply_data.get("score", 0),
                        "created_utc": reply_data.get("created_utc"),
                        "depth": 1,
                    }
                )

    _emit_progress(
        "Fetched post comments",
        subreddit=subreddit,
        post_id=post_id,
        extracted_comments=len(extracted),
    )
    return extracted


def scrape_subreddit_keywords(
    subreddit: str,
    keywords: Iterable[str],
    time_filters: Iterable[str] = ("year", "month"),
    pages: int = 1,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Run keyword search across time filters and return deduplicated posts."""
    all_posts: List[Dict[str, Any]] = []
    for keyword in keywords:
        for time_filter in time_filters:
            posts = search_subreddit_keyword(
                subreddit=subreddit,
                keyword=keyword,
                time_filter=time_filter,
                pages=pages,
                headers=headers,
            )
            all_posts.extend(posts)

    return deduplicate_posts(all_posts)
