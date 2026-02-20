"""Input/output helpers for local persistence and logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_output_dir(path: str | Path) -> Path:
    """Create output directory when needed."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_csv(frame: pd.DataFrame, output_path: str | Path) -> None:
    """Persist DataFrame to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def save_json(data: dict[str, Any] | list[Any], output_path: str | Path) -> None:
    """Persist JSON payload to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def incremental_jsonl_write(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Append records in JSON-lines format for incremental saves."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_log(message: str, log_path: str | Path) -> None:
    """Append simple timestamp-free log lines."""
    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
