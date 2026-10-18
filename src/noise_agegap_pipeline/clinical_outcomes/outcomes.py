from __future__ import annotations
import numpy as np
import pandas as pd
from ..common import require, unique_eid

SPECIAL_DATES = [
    "1900-01-01",
    "1901-01-01",
    "1902-02-02",
    "1903-03-03",
    "1909-09-09",
    "2037-07-07",
]


def ukb_dates(values):
    return pd.to_datetime(values.replace(SPECIAL_DATES, np.nan), errors="coerce")


def build_riskset(df, event_column, cfg):
    spec = cfg["clinical"]
    require(df, [spec["baseline_column"], event_column])
    unique_eid(df)
    baseline = ukb_dates(df[spec["baseline_column"]])
    event = ukb_dates(df[event_column])
    ends = pd.DataFrame(
        {"administrative": pd.Timestamp(spec["administrative_censor"])}, index=df.index
    )
    for col in spec["death_columns"]:
        if col in df:
            ends[col] = ukb_dates(df[col])
    if spec.get("loss_column"):
        require(df, [spec["loss_column"]])
        ends["loss"] = ukb_dates(df[spec["loss_column"]])
    censor = ends.min(axis=1)
    prevalent = event.notna() & baseline.notna() & (event <= baseline)
    incident = event.notna() & (event > baseline) & (event <= censor)
    stop = censor.mask(incident, event)
    duration = (stop - baseline).dt.days / 365.25
    valid = baseline.notna() & (~prevalent) & duration.notna() & (duration > 0)
    block = df.loc[valid].copy()
    block["time_years"] = duration.loc[valid]
    block["event"] = incident.loc[valid].astype(int)
    qc = {
        "input_n": len(df),
        "prevalent_excluded_n": int(prevalent.sum()),
        "riskset_n": len(block),
        "events": int(block.event.sum()),
        "baseline_column": spec["baseline_column"],
        "censor_date": spec["administrative_censor"],
    }
    return block, qc
