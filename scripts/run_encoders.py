"""Fine-tune BERT-family encoders and run them through every evaluation track.

Results are appended to results/<tag>/ next to the baselines (rows for other
methods are kept). Folds, splits and bootstrap use seed 0 as for the
baselines; --model-seed varies only the encoder's initialisation, data order
and inner validation split, so repeated seeds measure training variance.

    # Colab (GPU): both models, both heads, seed 0, into results/primary
    python -m scripts.run_encoders --models bioclinicalbert roberta --heads binary ordinal
    # extra seeds go to their own tags
    python -m scripts.run_encoders --model-seed 1 --tag seeds/s1
    # local smoke test (CPU/MPS, ~minutes): tiny model, short inputs
    python -m scripts.run_encoders --models tiny --heads binary --epochs 1 --max-len 128 --n-boot 50 --tag smoke
"""

from __future__ import annotations

import argparse

from triage import data, harness
from triage.data import ROOT
from triage.methods.encoder import ENCODERS, EncoderMethod

from .run_baselines import run_methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["bioclinicalbert", "roberta"], choices=list(ENCODERS))
    ap.add_argument("--heads", nargs="*", default=["binary", "ordinal"], choices=["binary", "ordinal"])
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-level", type=float, default=data.URGENT_MAX_LEVEL)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tag", default="primary")
    args = ap.parse_args()

    methods = {}
    for m in args.models:
        for h in args.heads:
            e = EncoderMethod(m, h, seed=args.model_seed, lr=args.lr, epochs=args.epochs,
                              batch_size=args.batch_size, max_len=args.max_len)
            methods[e.name] = e
    corpora = harness.Corpora.load(args.max_level)
    run_methods(methods, corpora, ROOT / "results" / args.tag, seed=0, n_boot=args.n_boot, append=True,
                config={**vars(args), "config_name": "encoders"})


if __name__ == "__main__":
    main()
