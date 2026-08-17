#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgae_v3.benchmark.policy_qualification import qualify_structural_policy
from lgae_v3.version import VERSION, QUALIFICATION_SCHEMA
from lgae_v3.reproducibility import ReproducibilityInfo, qualification_id


def main() -> int:
    repro = ReproducibilityInfo.collect(seed=42, repo=ROOT)
    qid = qualification_id(repro)
    _, result = qualify_structural_policy()
    payload = {
        "version": VERSION,
        "schema": QUALIFICATION_SCHEMA,
        "qualification_id": qid,
        "reproducibility": repro.to_dict(),
        "policy_qualification": result.to_dict(),
        "thresholds": {
            "diagnosis_accuracy_min": 0.80,
            "mean_regret_max": 0.35,
        },
    }
    payload["passed"] = (
        result.diagnosis_accuracy >= payload["thresholds"]["diagnosis_accuracy_min"]
        and result.mean_regret <= payload["thresholds"]["mean_regret_max"]
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
