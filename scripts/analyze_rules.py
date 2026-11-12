"""Which rule categories fire on urgent vs non-urgent calibration messages."""

import pandas as pd

from triage import data
from triage.data import ROOT
from triage.methods.rules import CATEGORIES, RulesLR, hit_matrix

pool = data.load_development()
h = pd.DataFrame(hit_matrix(pool["text"]), columns=list(CATEGORIES))
y = pool["urgent"].values
w = RulesLR().fit(pool["text"], pool["level"], y).weights()
tab = pd.DataFrame(
    {
        "tier": [CATEGORIES[c][0] for c in h],
        "urgent_hit_rate": h[y].mean(),
        "nonurgent_hit_rate": h[~y].mean(),
        "n_hits": h.sum().astype(int),
        "lr_weight": pd.Series(w),
    }
).sort_values(["tier", "urgent_hit_rate"], ascending=[True, False])
tab["lift"] = tab["urgent_hit_rate"] / tab["nonurgent_hit_rate"].clip(lower=1e-3)
out = ROOT / "results" / "primary" / "rule_coverage.csv"
tab.round(3).to_csv(out)
print(tab.round(3).to_string())
none = (h.sum(1) == 0)
print(f"\nurgent messages with no rule hit: {none[y].mean():.2%} ({none[y].sum()}/{y.sum()})")
