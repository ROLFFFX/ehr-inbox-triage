from .rules import Keyword5110, RulesAny, RulesLR, RulesTiered
from .tfidf import FilterCascade, KeywordBypass, TfidfLR, TfidfOrdinal, TfidfRulesLR


def registry(seed: int = 0) -> dict:
    """Methods available up to the Week 3 milestone (before encoders)."""
    methods = [
        Keyword5110(),
        RulesAny(),
        RulesTiered(),
        RulesLR(),
        TfidfLR(seed),
        TfidfOrdinal(),
        TfidfRulesLR(seed=seed),
        KeywordBypass(TfidfLR(seed)),
        FilterCascade(seed),
    ]
    return {m.name: m for m in methods}
