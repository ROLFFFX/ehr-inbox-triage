"""Dataset loaders.

Every loader returns a DataFrame with a common schema:

    msg_id  : stable identifier, "<source>:<split>:<index>"
    text    : message text the methods see
    level   : PMR ordinal urgency, 1 = emergency ... 6 = no medical attention.
              May be fractional on PMR-Reddit (aggregated over multiple
              clinician responses). NaN where a corpus has no PMR level.
    urgent  : binary target for the classification track
    source  : corpus name
    split   : split name within the corpus

Raw files are fetched by scripts/download_data.py into data/raw/.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The author keeps data and the report in a git-ignored private/ folder;
# a fresh clone uses data/ and report/ at the top level.
_LOCAL = ROOT / "private"
RAW = (_LOCAL if (_LOCAL / "data").exists() else ROOT) / "data" / "raw"
REPORT = (_LOCAL if (_LOCAL / "report").exists() else ROOT) / "report"

# Primary urgency definition: PMR Level 1 ("emergency attention needed").
# Fractional labels up to 1.5 mean a majority of responses put the message at
# Level 1, so they count as urgent.
URGENT_MAX_LEVEL = 1.5

# C-SSRS severity (0-6) at or above which an r/SuicideWatch post counts as
# urgent: active ideation with intent (4), with plan (5), or behaviour (6).
CSSRS_URGENT_MIN = 4

HF_DATASETS = {
    "PMR-Reddit-Train-Dev-Pool": "PortalPal-AI/PMR-Reddit-Train-Dev-Pool",
    "PMR-Reddit-Test-Pairs": "PortalPal-AI/PMR-Reddit-Test-Pairs",
    "PMR-Synth-Inboxes": "PortalPal-AI/PMR-Synth-Inboxes",
    "CSSR-S_labelled_suicidewatch_posts_reddit": "av9ash/CSSR-S_labelled_suicidewatch_posts_reddit",
}


def _finish(df: pd.DataFrame, source: str, split: str, max_level: float = URGENT_MAX_LEVEL) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["msg_id"] = [f"{source}:{split}:{i}" for i in range(len(df))]
    df["source"] = source
    df["split"] = split
    if "urgent" not in df:
        df["urgent"] = df["level"] <= max_level
    return df[["msg_id", "text", "level", "urgent", "source", "split"]]


def load_reddit_pool(max_level: float = URGENT_MAX_LEVEL) -> pd.DataFrame:
    """PMR-Reddit train (772) and dev (87) message pools."""
    base = RAW / "PMR-Reddit-Train-Dev-Pool" / "data"
    parts = []
    for split in ("train", "dev"):
        d = pd.read_parquet(base / f"{split}-00000-of-00001.parquet")[["text", "level"]]
        parts.append(_finish(d, "pmr_reddit", split, max_level))
    return pd.concat(parts, ignore_index=True)


def load_reddit_test_pairs() -> pd.DataFrame:
    """The 1,502 official pairwise test comparisons (chosen = more urgent)."""
    return pd.read_parquet(RAW / "PMR-Reddit-Test-Pairs" / "data" / "test-00000-of-00001.parquet")


def load_reddit_test(max_level: float = URGENT_MAX_LEVEL) -> pd.DataFrame:
    """The 362 unique messages underlying the official test pairs."""
    pairs = load_reddit_test_pairs()
    a = pairs[["chosen", "chosen_level"]].set_axis(["text", "level"], axis=1)
    b = pairs[["rejected", "rejected_level"]].set_axis(["text", "level"], axis=1)
    msgs = pd.concat([a, b]).drop_duplicates("text").sort_values("text", kind="stable")
    return _finish(msgs, "pmr_reddit", "test", max_level)


_PATIENT_MSG = re.compile(r"###\s*Patient Message:\s*", re.IGNORECASE)


def _synth_message(text: str) -> str:
    """Keep only the patient message; drop the prepended EHR summary."""
    parts = _PATIENT_MSG.split(text, maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else text.strip()


def load_synth(with_ehr: bool = False, max_level: float = URGENT_MAX_LEVEL) -> pd.DataFrame:
    """PMR-Synth: two expert-labelled 30-message inboxes with EHR context."""
    base = RAW / "PMR-Synth-Inboxes" / "data"
    parts = []
    for split in ("INBOX_A", "INBOX_B"):
        d = pd.read_parquet(base / f"{split}-00000-of-00001.parquet")
        d = d.assign(level=d["level"].astype(float))
        if not with_ehr:
            d = d.assign(text=d["text"].map(_synth_message))
        parts.append(_finish(d[["text", "level"]], "pmr_synth", split, max_level))
    return pd.concat(parts, ignore_index=True)


def load_cssrs() -> pd.DataFrame:
    """r/SuicideWatch posts with psychologist-assigned C-SSRS severity (0-6).

    Used as an out-of-domain psychiatry stress test. `level` is left NaN
    because C-SSRS severity is not on the PMR scale; `cssrs` keeps it.
    """
    d = pd.read_csv(RAW / "CSSR-S_labelled_suicidewatch_posts_reddit" / "labeled_rSuicidewatch_posts.csv")
    d = d.dropna(subset=["content", "severity"])
    out = pd.DataFrame(
        {
            "text": d["content"].astype(str).values,
            "level": np.nan,
            "urgent": (d["severity"].astype(int) >= CSSRS_URGENT_MIN).values,
        }
    )
    out = _finish(out, "cssrs_suicidewatch", "all")
    out["cssrs"] = d["severity"].astype(int).values
    return out


def load_development(max_level: float = URGENT_MAX_LEVEL) -> pd.DataFrame:
    """Train + dev pool: the only data used for fitting and threshold selection."""
    return load_reddit_pool(max_level)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["source", "split"], sort=False)
    return pd.DataFrame(
        {
            "n": g.size(),
            "n_urgent": g["urgent"].sum().astype(int),
            "prevalence": g["urgent"].mean().round(3),
            "median_chars": g["text"].apply(lambda s: int(s.str.len().median())),
        }
    )
