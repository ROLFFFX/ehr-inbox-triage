"""Rule-based urgency scorers.

The lexicon is organised by the symptom categories that nurse telephone-triage
protocols route to emergency or same-day dispositions (chest pain, breathing
difficulty, stroke signs, suicidality, ...). Patterns are written by hand
for this project and listed in full here so the lexicon is auditable; they
are not copied from a licensed protocol book.

Tier 1 = emergency disposition (call 911 / go to ED now).
Tier 2 = same-day disposition (urgent care / see today).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

# fmt: off
CATEGORIES: dict[str, tuple[int, list[str]]] = {
    # ---- Tier 1: emergency -------------------------------------------------
    "suicidality": (1, [
        r"suicid\w*", r"kill(ing)? my ?self", r"end(ing)? (my|it all|my life)", r"take my (own )?life",
        r"(don'?t|do not) want to (live|be alive|wake up)", r"better off dead", r"want(ed)? to die",
        r"no reason to live", r"(plan|planning) to (die|end)", r"goodbye (note|letter)",
    ]),
    "self_harm": (1, [
        r"self[- ]?harm\w*", r"cut(ting)? my(self| wrists?| arms?)", r"hurt(ing)? my ?self",
        r"burn(ing|ed)? my ?self",
    ]),
    "harm_to_others": (1, [
        r"homicid\w*", r"kill (him|her|them|someone|people)", r"hurt (someone|somebody|others|him|her)",
        r"violent thoughts",
    ]),
    "overdose_poisoning": (1, [
        r"overdos\w*", r"\bOD'?(ed|d)?\b", r"took (too many|a bunch of|all (of )?my|the whole)",
        r"poison\w*", r"swallow(ed)? (bleach|chemicals?|batter(y|ies))", r"ingest(ed)?",
    ]),
    "psychosis_acute": (1, [
        r"command hallucinations?", r"voices (telling|tell) me", r"hearing voices", r"psychotic break",
        r"paranoi\w* (that|they)",
    ]),
    "chest_pain": (1, [
        r"chest (pain|pressure|tightness|tight)", r"pain in (my|the) chest", r"heart attack",
        r"crushing (pain|feeling)", r"pain (radiating|spreading) (to|down) (my )?(left )?arm",
    ]),
    "breathing": (1, [
        r"can'?t breathe", r"cannot breathe", r"(trouble|difficulty|hard|struggling) (to )?breath\w*",
        r"short(ness)? of breath", r"gasping", r"choking", r"lips? (turning )?(blue|purple)",
        r"wheez\w*", r"\bsp ?o2\b", r"oxygen (level|sat)",
    ]),
    "stroke_neuro": (1, [
        r"stroke", r"face (is )?droop\w*", r"drooping", r"slurred speech|slurring",
        r"(sudden|suddenly) (weak|numb|confus|vision|blind)\w*", r"one side of (my|the) (body|face)",
        r"worst headache", r"thunderclap",
    ]),
    "loss_of_consciousness": (1, [
        r"pass(ed|ing)? out", r"faint(ed|ing)", r"lost consciousness|loss of consciousness",
        r"unconscious", r"unresponsive", r"black(ed)? out",
    ]),
    "seizure": (1, [r"seiz(ure|ures|ing)", r"convuls\w*"]),
    "severe_bleeding": (1, [
        r"(won'?t|will not|can'?t) stop bleeding", r"bleeding (heavily|a lot|profusely)",
        r"(vomit|throw)\w* (up )?blood", r"cough\w* (up )?blood", r"blood in (my )?(vomit|stool)",
        r"black,? tarry", r"hemorrhag\w*",
    ]),
    "anaphylaxis": (1, [
        r"anaphyla\w*", r"throat (is )?(closing|swelling|swollen)", r"tongue (is )?swell\w*",
        r"swollen (lips|tongue|face)", r"epi ?pen",
    ]),
    "pregnancy_emergency": (1, [
        r"pregnan\w*.{0,60}(bleed|cramp|pain|fluid)", r"(bleed|cramp|pain)\w*.{0,60}pregnan\w*",
        r"water broke", r"(decreased|reduced|no) fetal movement", r"baby (isn'?t|not) moving",
    ]),
    "meningitis_sepsis": (1, [
        r"stiff neck", r"neck (is )?stiff", r"rash.{0,40}(fever|won'?t fade)", r"septic|sepsis",
    ]),
    "severe_abdominal": (1, [
        r"(severe|excruciating|worst|unbearable) (abdominal|stomach|belly) pain",
        r"(abdominal|stomach) pain.{0,40}(severe|excruciating|unbearable)", r"rigid (abdomen|stomach)",
        r"appendicitis",
    ]),
    "trauma": (1, [
        r"hit (my|his|her) head", r"head (injury|trauma)", r"concussion", r"car (accident|crash)",
        r"\bfell\b.{0,40}(head|hit)", r"bone (sticking|poking) out",
    ]),
    # ---- Tier 2: same-day -------------------------------------------------
    "high_fever": (2, [
        r"fever of (10[3-9]|39|4[0-2])", r"\b10[3-5](\.\d)? ?(f|degrees)", r"high fever",
    ]),
    "severe_pain": (2, [
        r"(severe|excruciating|unbearable|worst) pain", r"pain (is )?(10|9)/10", r"(10|9) out of 10",
    ]),
    "infection_spread": (2, [
        r"red streak", r"spreading (redness|rash)", r"(swollen|hot) and red", r"pus",
        r"infect(ed|ion).{0,40}(spread|worse)",
    ]),
    "dehydration": (2, [
        r"can'?t keep (anything|food|water|fluids) down", r"haven'?t (peed|urinated)", r"dehydrat\w*",
    ]),
    "vision_change": (2, [r"sudden(ly)? (lost|loss of|blurry|blurred|double) vision", r"vision loss"]),
    "testicular": (2, [r"testic\w* (pain|torsion|swelling)", r"(sudden|severe) (pain in )?(my )?(testicle|scrotum)"]),
    "mental_health_crisis": (2, [
        r"panic attack", r"can'?t stop (crying|shaking)", r"(manic|mania)\b", r"haven'?t slept in (\d+|days)",
        r"(severe|extreme) (anxiety|depression)", r"breakdown", r"withdraw(al|ing)",
    ]),
}
# fmt: on

# The literal keyword-bypass list in the BMIN 5110 data model ("harm",
# "overdose", "suicidal"), used to test the group's own safety-floor design.
KEYWORDS_5110 = [r"harm", r"overdos\w*", r"suicid\w*"]

NEGATION = re.compile(
    r"\b(no|not|never|without|denies|deny|denied|don'?t|doesn'?t|didn'?t|isn'?t|wasn'?t|haven'?t|hasn'?t|nor)\b",
    re.IGNORECASE,
)
NEG_WINDOW = 5  # tokens before a match that can negate it


def _compile(patterns: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(patterns) + r")", re.IGNORECASE)


COMPILED = {name: (tier, _compile(pats)) for name, (tier, pats) in CATEGORIES.items()}


def _negated(text: str, start: int) -> bool:
    prefix = re.split(r"[.!?\n]", text[:start])[-1]  # same sentence only
    return bool(NEGATION.search(" ".join(prefix.split()[-NEG_WINDOW:])))


def category_hits(text: str, use_negation: bool = True) -> dict[str, int]:
    """1 if any non-negated pattern of the category fires, else 0."""
    hits = {}
    for name, (_, pat) in COMPILED.items():
        hits[name] = int(any(not (use_negation and _negated(text, m.start())) for m in pat.finditer(text)))
    return hits


def hit_matrix(texts, use_negation: bool = True) -> np.ndarray:
    return np.array([list(category_hits(t, use_negation).values()) for t in texts], dtype=float)


TIERS = np.array([tier for tier, _ in CATEGORIES.values()])


@dataclass
class Keyword5110:
    """Binary bypass on the three 5110 keywords (no negation handling)."""

    name: str = "kw5110"
    nominal_threshold: float = 0.5
    needs_fit: bool = False

    def score(self, texts) -> np.ndarray:
        pat = _compile(KEYWORDS_5110)
        return np.array([float(bool(pat.search(t))) for t in texts])


@dataclass
class RulesAny:
    """1 if any tier-1 category fires (negation-aware), else 0."""

    name: str = "rules_any"
    nominal_threshold: float = 0.5
    needs_fit: bool = False

    def score(self, texts) -> np.ndarray:
        return (hit_matrix(texts)[:, TIERS == 1].sum(1) > 0).astype(float)


@dataclass
class RulesTiered:
    """Ordinal score: 2 per tier-1 category + 1 per tier-2 category."""

    name: str = "rules_tiered"
    nominal_threshold: float = 0.5
    needs_fit: bool = False
    use_negation: bool = True

    def score(self, texts) -> np.ndarray:
        h = hit_matrix(texts, self.use_negation)
        return h @ np.where(TIERS == 1, 2.0, 1.0)


class RulesLR:
    """Rule category hits as features, weights learned by logistic regression.

    This is the "tuned" rule baseline: the lexicon is fixed, only the
    per-category weights are fit on training data.
    """

    name = "rules_lr"
    needs_fit = True
    nominal_threshold = 0.0  # P(urgent) = 0.5

    def __init__(self, C: float = 1.0):
        self.model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)

    def fit(self, texts, level, urgent):
        self.model.fit(hit_matrix(texts), urgent)
        return self

    def score(self, texts) -> np.ndarray:
        return self.model.decision_function(hit_matrix(texts))

    def weights(self) -> dict[str, float]:
        return dict(zip(CATEGORIES, self.model.coef_[0]))
