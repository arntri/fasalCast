"""
Single source of truth for which (market_id, commodity_id) pairs are viable.

Decision (interview-defensible): baseline-gating only means something if the
baseline and every ML model face the SAME pair population. A pair that failed
validate.py's data-quality gates (coverage / min-rows) produces unreliable
scores, so it is excluded from the evaluation harness entirely — for the
baseline as much as for any model. This restriction is parsed ONCE here so it
cannot drift between features.py and evaluate.py.

The verdict comes from reports/quality_report.json ("VIABLE_FOR_MODELING"),
never hardcoded. The report keys look like "market_169 | Onion"; the commodity
NAME is mapped back to its id via config.yaml's commodity_names so new
commodities (Tomato/Potato) join automatically once config is updated.
"""

import json
from pathlib import Path

import yaml


def viable_pairs(report_path="reports/quality_report.json",
                 config_path="config.yaml"):
    """Return a sorted list of (market_id, commodity_id) tuples marked
    VIABLE_FOR_MODELING in the quality report. Nothing hardcoded."""
    rep = json.loads(Path(report_path).read_text())
    cfg = yaml.safe_load(open(config_path))
    name_to_id = {str(v).lower(): int(k)
                  for k, v in (cfg.get("commodity_names") or {23: "Onion"}).items()}

    out = []
    for key, r in rep.get("pairs", {}).items():
        if not r.get("VIABLE_FOR_MODELING"):
            continue
        mkt_part, _, com_part = key.partition("|")
        mkt_part, com_part = mkt_part.strip(), com_part.strip()
        try:
            market_id = int(mkt_part.replace("market_", ""))
        except ValueError:
            print(f"  (pairs: could not parse market id from '{key}', skipping)")
            continue
        commodity_id = name_to_id.get(com_part.lower())
        if commodity_id is None:
            print(f"  (pairs: '{com_part}' not in config commodity_names — "
                  f"add it so this viable pair can join the pool; skipping)")
            continue
        out.append((market_id, commodity_id))
    return sorted(set(out))


if __name__ == "__main__":
    print("Viable (market_id, commodity_id) pairs:", viable_pairs())
