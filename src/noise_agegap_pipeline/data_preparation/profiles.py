from __future__ import annotations
import pandas as pd
from ..common import unique_eid, require
from ..age_inference.inference import validate_profile


def build_profiles(frame, profile_fields, excluded_noise_fields, max_missing=0.60):
    """Construct structured profiles from an explicit author-approved field whitelist.

    Completeness is computed on this whitelist, not all UKB columns. This function
    cannot infer the historical feature denominator from a processed analysis CSV.
    """
    if not profile_fields:
        raise ValueError("Profile whitelist must not be empty")
    unique_eid(frame)
    require(frame, ["chronological_age"] + profile_fields)
    rows = []
    audit = []
    for case in frame.to_dict("records"):
        missing = sum(pd.isna(case[c]) for c in profile_fields)
        excluded = missing / len(profile_fields) > max_missing
        audit.append(
            {
                "eid": case["eid"],
                "feature_n": len(profile_fields),
                "missing_n": missing,
                "excluded_gt_60pct": excluded,
            }
        )
        if excluded:
            continue
        profile = {c: case[c] for c in profile_fields if pd.notna(case[c])}
        validate_profile(profile, excluded_noise_fields)
        rows.append(
            {
                "eid": case["eid"],
                "chronological_age": case["chronological_age"],
                "profile": profile,
            }
        )
    return rows, pd.DataFrame(audit)
