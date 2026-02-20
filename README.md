# Reddit Social Listening (Skeleton)

A starter Python project for collecting Reddit posts, running lightweight sentiment/topic analysis, and producing report-ready output files.

## Project layout

```text
.
├── pyproject.toml
├── README.md
└── src/
    └── reddit_social_listening/
        ├── __init__.py
        ├── config.py
        ├── scraper.py
        ├── analyze.py
        ├── aggregate.py
        ├── io_utils.py
        └── main.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

Using the installed console script:

```bash
reddit-social-listening --output-dir output --max-pages 3
```

Or with Python module execution:

```bash
python -m reddit_social_listening.main --output-dir output --max-pages 3
```

## Output files

The pipeline writes files into `output/` by default:

- `posts_analyzed.csv` - row-level analyzed posts
- `topic_summary.csv` - topic mention counts
- `bubble_chart_data.csv` - topic volume + average sentiment
- `posts_analyzed.jsonl` - incremental analyzed rows
- `run_metadata.json` - run details and generated file list
- `pipeline.log` - basic run log lines

## Expected runtime

Runtime depends on the number of subreddits/pages and your network conditions.
For the default configuration (3 subreddits, 3 pages each), expect roughly **1-5 minutes**.
