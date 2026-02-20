"""Configuration for the Reddit social listening pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class PipelineConfig:
    """Settings used by the scraper and analysis pipeline."""

    subreddits: list[str] = field(default_factory=lambda: ["technology", "startups", "gadgets"])
    keywords: list[str] = field(default_factory=lambda: ["brand", "product", "launch", "pricing"])
    min_upvotes: int = 5
    min_comments: int = 1
    date_cutoff: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    )

    request_timeout_s: int = 20
    requests_per_second: float = 1.0
    max_retries: int = 3
    backoff_base_s: float = 1.5


DEFAULT_CONFIG = PipelineConfig()
