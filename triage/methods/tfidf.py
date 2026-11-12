"""Classical ML baselines on TF-IDF features."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from .rules import KEYWORDS_5110, _compile, hit_matrix

C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ]
    )


def _tuned_lr(seed: int) -> GridSearchCV:
    pipe = Pipeline(
        [("feats", _features()), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]
    )
    # C is chosen by inner CV on average precision, so the outer split never
    # sees tuning decisions made on its own held-out fold.
    return GridSearchCV(
        pipe,
        {"clf__C": C_GRID},
        scoring="average_precision",
        cv=StratifiedKFold(4, shuffle=True, random_state=seed),
        n_jobs=-1,
    )


class TfidfLR:
    """Word + character n-gram TF-IDF with class-balanced logistic regression."""

    name = "tfidf_lr"
    needs_fit = True
    nominal_threshold = 0.0  # P(urgent) = 0.5

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, texts, level, urgent):
        self.model = _tuned_lr(self.seed).fit(list(texts), urgent)
        return self

    def score(self, texts) -> np.ndarray:
        return self.model.decision_function(list(texts))


class TfidfOrdinal:
    """Ridge regression on the PMR level; score = negative predicted level.

    Uses the full ordinal label rather than the binary cut, which can help
    the ranking track.
    """

    name = "tfidf_ordinal"
    needs_fit = True

    def __init__(self, alpha: float = 3.0):
        self.model = Pipeline([("feats", _features()), ("reg", Ridge(alpha=alpha))])

    def fit(self, texts, level, urgent):
        self.model.fit(list(texts), level)
        return self

    def score(self, texts) -> np.ndarray:
        return -self.model.predict(list(texts))


class TfidfRulesLR:
    """TF-IDF features stacked with rule-category hits (hybrid)."""

    name = "tfidf+rules_lr"
    needs_fit = True
    nominal_threshold = 0.0

    def __init__(self, C: float | None = None, rule_scale: float = 1.0, seed: int = 0):
        self.C, self.rule_scale, self.seed = C, rule_scale, seed

    def _x(self, texts, fit=False):
        texts = list(texts)
        tf = self.feats.fit_transform(texts) if fit else self.feats.transform(texts)
        return hstack([tf, csr_matrix(self.rule_scale * hit_matrix(texts))]).tocsr()

    def fit(self, texts, level, urgent):
        self.feats = _features()
        x = self._x(texts, fit=True)
        search = GridSearchCV(
            LogisticRegression(class_weight="balanced", max_iter=5000),
            {"C": C_GRID if self.C is None else [self.C]},
            scoring="average_precision",
            cv=StratifiedKFold(4, shuffle=True, random_state=self.seed),
            n_jobs=-1,
        )
        self.model = search.fit(x, urgent)
        return self

    def score(self, texts) -> np.ndarray:
        return self.model.decision_function(self._x(texts))


class KeywordBypass:
    """The 5110 safety-floor design: keyword match forces the top score,
    otherwise defer to the wrapped model's score."""

    needs_fit = True

    def __init__(self, base):
        self.base = base
        self.name = f"{base.name}+bypass"
        self.nominal_threshold = getattr(base, "nominal_threshold", None)
        self.pat = _compile(KEYWORDS_5110)

    def fit(self, texts, level, urgent):
        self.base.fit(texts, level, urgent)
        return self

    def score(self, texts) -> np.ndarray:
        s = self.base.score(texts)
        hit = np.array([bool(self.pat.search(t)) for t in texts])
        return np.where(hit, 1e6, s)  # constant so calibration and test agree


class FilterCascade:
    """Two-stage cascade after Swaminathan et al. (2023): a broad keyword
    filter (any rule category, tier 1 or 2, no negation) gates a TF-IDF LR.
    Messages that fail the filter are ranked below every message that passes.
    """

    name = "cascade_filter_lr"
    needs_fit = True
    nominal_threshold = 1.5  # passes filter and P > 0.5

    def __init__(self, seed: int = 0):
        self.lr = TfidfLR(seed)

    @staticmethod
    def _pass(texts) -> np.ndarray:
        return hit_matrix(texts, use_negation=False).sum(1) > 0

    def fit(self, texts, level, urgent):
        # Fit stage 2 on all data: with ~90 positives, training only on
        # filter-passing messages leaves too few examples.
        self.lr.fit(texts, level, urgent)
        return self

    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        s = self.lr.score(texts)
        p = 1.0 / (1.0 + np.exp(-s))
        return np.where(self._pass(texts), 1.0 + p, p - 1.0)
