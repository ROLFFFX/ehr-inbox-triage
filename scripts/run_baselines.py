"""Run the pre-encoder methods through both evaluation tracks.

    python -m scripts.run_baselines                 # primary definition (Level 1)
    python -m scripts.run_baselines --max-level 2   # sensitivity: Levels 1-2
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from triage import data, harness
from triage.data import ROOT
from triage.methods import registry


def _merge(path, rows: list[dict], append: bool):
    """Write result rows; with append, replace only the rows for these methods."""
    new = pd.DataFrame(rows)
    if append and path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["method"].isin(new["method"])], new], ignore_index=True)
    new.to_csv(path, index=False)


def run_methods(methods: dict, corpora: harness.Corpora, out, seed: int = 0, n_boot: int = 2000,
                append: bool = False, config: dict | None = None):
    (out / "scores").mkdir(parents=True, exist_ok=True)
    desc = pd.concat([data.describe(d) for d in (corpora.pool, corpora.test, corpora.synth, corpora.cssrs)])
    desc.to_csv(out / "corpora.csv")
    print(desc.to_string(), "\n")

    cls, rnk, trn = [], [], []
    for name, method in methods.items():
        t0 = time.time()
        r = harness.evaluate(method, corpora, seed=seed, n_boot=n_boot)
        cls.append({"method": name, **r.classification})
        rnk.append({"method": name, **r.ranking})
        trn.append({"method": name, **r.transfer})
        for split, df in r.scores.items():
            df.to_parquet(out / "scores" / f"{name}__{split}.parquet")
        r.inboxes.to_parquet(out / "scores" / f"{name}__inboxes.parquet")
        c = r.classification
        print(
            f"{name:22s} recall={c['recall']:.3f} [{c['recall_lo']:.2f},{c['recall_hi']:.2f}] "
            f"FA/100={c['fa_per_100']:5.1f} [{c['fa_per_100_lo']:.1f},{c['fa_per_100_hi']:.1f}] "
            f"AUROC={c['auroc']:.3f} R@FA10={c['recall_at_fa10']:.3f} R@PPV67={c['recall_at_ppv67']:.3f} "
            f"T-NDCG@10={r.ranking['sim_t_ndcg@10']:.3f} pairs={r.ranking['pair_acc_all']:.3f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        # Write after every method so a dropped Colab session keeps finished work.
        _merge(out / "classification.csv", cls, append=True)
        _merge(out / "ranking.csv", rnk, append=True)
        _merge(out / "transfer.csv", trn, append=True)

    if not append:  # a fresh run replaces the files wholesale
        _merge(out / "classification.csv", cls, append=False)
        _merge(out / "ranking.csv", rnk, append=False)
        _merge(out / "transfer.csv", trn, append=False)
    if config is not None:
        (out / f"config{'_' + config['config_name'] if 'config_name' in config else ''}.json").write_text(
            json.dumps(config, indent=2)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", help="subset of method names")
    ap.add_argument("--max-level", type=float, default=data.URGENT_MAX_LEVEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--append", action="store_true", help="keep rows for other methods already in results/<tag>")
    args = ap.parse_args()

    corpora = harness.Corpora.load(args.max_level)
    methods = registry(args.seed)
    names = args.methods or list(methods)
    run_methods({n: methods[n] for n in names}, corpora, ROOT / "results" / args.tag, seed=args.seed,
                n_boot=args.n_boot, append=args.append, config=vars(args))


if __name__ == "__main__":
    main()
