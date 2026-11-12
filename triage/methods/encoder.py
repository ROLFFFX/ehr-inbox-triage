"""Fine-tuned transformer encoders (BERT family).

Same interface as the TF-IDF methods: ``fit(texts, level, urgent)`` and
``score(texts)``, so the harness handles CV calibration, threshold locking
and both tracks unchanged.

Hyperparameters are fixed in advance (no search on any evaluation data):
the epoch is chosen on a stratified 10% split held out inside each training
fold. Two heads share one regression output:

- ``binary``:  BCE on Level-1 vs rest, positives up-weighted by the class
  ratio; score = logit (nominal decision rule: logit > 0, i.e. P > 0.5).
- ``ordinal``: MSE on the standardised PMR level; score = -predicted level
  (nominal rule: predicted level <= 1.5).
"""

from __future__ import annotations

import copy
import random

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

# Hugging Face IDs. "tiny" is for smoke tests only.
ENCODERS = {
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "roberta": "roberta-base",
    "tiny": "google/bert_uncased_L-2_H-128_A-2",
}
LEVEL_MEAN, LEVEL_SD = 3.5, 1.7  # rough PMR-level scale, keeps the regression target near unit variance


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EncoderMethod:
    needs_fit = True

    def __init__(
        self,
        model_key: str = "roberta",
        head: str = "binary",
        seed: int = 0,
        lr: float = 2e-5,
        epochs: int = 5,
        batch_size: int = 16,
        max_len: int = 512,
        warmup: float = 0.1,
        weight_decay: float = 0.01,
        val_frac: float = 0.1,
    ):
        assert head in ("binary", "ordinal")
        self.model_key, self.model_id, self.head = model_key, ENCODERS[model_key], head
        self.seed, self.lr, self.epochs, self.batch_size = seed, lr, epochs, batch_size
        self.max_len, self.warmup, self.weight_decay, self.val_frac = max_len, warmup, weight_decay, val_frac
        self.name = f"{model_key}_{head}"
        self.nominal_threshold = 0.0 if head == "binary" else -1.5
        self.model = self.tokenizer = None
        self.best_epoch = None

    def __deepcopy__(self, memo):
        # The harness deep-copies an unfitted method per fold; never copy weights.
        fresh = copy.copy(self)
        fresh.model = None
        fresh.best_epoch = None
        return fresh

    # ------------------------------------------------------------------
    def _batches(self, texts, idx, shuffle, rng=None):
        idx = np.array(idx)
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            b = idx[i : i + self.batch_size]
            enc = self.tokenizer([texts[j] for j in b], truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors="pt")
            yield b, enc

    def _predict(self, texts) -> np.ndarray:
        import torch

        texts = list(texts)
        dev = _device()
        self.model.to(dev).eval()
        # Sort by length so each batch pads to similar lengths.
        order = np.argsort([len(t) for t in texts])
        out = np.empty(len(texts), dtype=np.float32)
        with torch.no_grad():
            for b, enc in self._batches(texts, order, shuffle=False):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    logits = self.model(**{k: v.to(dev) for k, v in enc.items()}).logits.squeeze(-1)
                out[b] = logits.float().cpu().numpy()
        self.model.to("cpu")
        return out

    def fit(self, texts, level, urgent):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _seed_everything(self.seed)
        texts, level, urgent = list(texts), np.asarray(level, float), np.asarray(urgent, bool)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=1)
        dev = _device()
        self.model.to(dev)

        split = StratifiedShuffleSplit(1, test_size=self.val_frac, random_state=self.seed)
        tr, va = next(split.split(np.zeros(len(texts)), urgent))
        if self.head == "binary":
            target = urgent.astype(np.float32)
            pos_weight = torch.tensor(float((~urgent[tr]).sum() / max(urgent[tr].sum(), 1)), dtype=torch.float32, device=dev)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            target = ((level - LEVEL_MEAN) / LEVEL_SD).astype(np.float32)
            loss_fn = torch.nn.MSELoss()

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = self.epochs * int(np.ceil(len(tr) / self.batch_size))
        warm = max(1, int(self.warmup * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm)))
        )
        scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
        rng = np.random.default_rng(self.seed)

        best, best_state = -np.inf, None
        for epoch in range(1, self.epochs + 1):
            self.model.to(dev).train()
            for b, enc in self._batches(texts, tr, shuffle=True, rng=rng):
                y = torch.tensor(target[b], device=dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    logits = self.model(**{k: v.to(dev) for k, v in enc.items()}).logits.squeeze(-1)
                loss = loss_fn(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            # Epoch selection on the inner held-out split only.
            pv = self._predict([texts[i] for i in va])
            if self.head == "binary":
                metric = average_precision_score(urgent[va], pv)
            else:
                metric = -float(np.mean((pv - target[va]) ** 2))
            if metric > best:
                best, self.best_epoch = metric, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        return self

    def score(self, texts) -> np.ndarray:
        s = self._predict(texts)
        if self.head == "ordinal":
            # Higher = more urgent, on the PMR-level scale.
            s = -(s * LEVEL_SD + LEVEL_MEAN)
        return s
