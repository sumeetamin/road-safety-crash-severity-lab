"""Time-split baseline for multiple-fatality classification within fatal crashes."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "STATE", "MONTH", "DAY_WEEK", "HOUR", "RUR_URB", "FUNC_SYS", "ROUTE",
    "RELJCT1", "RELJCT2", "TYP_INT", "REL_ROAD", "WRK_ZONE", "LGT_COND",
    "WEATHER", "SCH_BUS", "RAIL",
]


def main():
    source = Path("data/accident.csv")
    df = pd.read_csv(source, low_memory=False)
    missing = [c for c in FEATURES + ["FATALS"] if c not in df]
    if missing:
        raise SystemExit(f"Missing expected FARS columns: {missing}")

    # Outcome is defined only inside FARS's fatal-crash census.
    d = df[FEATURES + ["FATALS"]].copy()
    for col in FEATURES:
        d[col] = d[col].astype("string").replace({"": pd.NA, "nan": pd.NA})
    d["MONTH"] = pd.to_numeric(d["MONTH"], errors="coerce")
    d["FATALS"] = pd.to_numeric(d["FATALS"], errors="coerce")
    d = d.dropna(subset=["MONTH", "FATALS"])
    d["multiple_fatalities"] = (d["FATALS"] >= 2).astype(int)
    train, test = d[d.MONTH < 10], d[d.MONTH >= 10]
    if train.multiple_fatalities.nunique() < 2 or test.multiple_fatalities.nunique() < 2:
        raise SystemExit("Both chronological partitions must contain both outcome classes.")

    # FARS special codes remain categories; no post-crash counts or injury fields enter X.
    x_cols = [c for c in FEATURES if c != "MONTH"] + ["MONTH"]
    prep = ColumnTransformer([
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]), x_cols)
    ])
    model = Pipeline([
        ("features", prep),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear")),
    ])
    model.fit(train[x_cols], train.multiple_fatalities)
    proba = model.predict_proba(test[x_cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y = test.multiple_fatalities
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    groups = {}
    for code, g in test.assign(_y=y.to_numpy(), _pred=pred).groupby("RUR_URB", dropna=False):
        groups[str(code)] = {
            "n": int(len(g)),
            "prevalence": float(g._y.mean()),
            "recall_at_0_5": float(recall_score(g._y, g._pred, zero_division=0)),
            "false_positive_rate_at_0_5": float(((g._pred == 1) & (g._y == 0)).sum() / max(1, (g._y == 0).sum())),
        }

    report = {
        "dataset": "NHTSA FARS 2023 National final release, accident.csv",
        "estimand": "Among crashes in the FARS fatal-crash census, classify whether at least two people died.",
        "rows": {"eligible": int(len(d)), "train_jan_sep": int(len(train)), "test_oct_dec": int(len(test))},
        "target_prevalence": {
            "train": float(train.multiple_fatalities.mean()),
            "test": float(y.mean()),
        },
        "model": "Class-weighted logistic regression, one-hot encoded context features",
        "features": x_cols,
        "excluded": ["ST_CASE", "FATALS", "PERSONS", "PERMVIT", "PEDS", "PERNOTMVIT", "coordinates", "post-crash response/injury fields"],
        "test_metrics_at_threshold_0_5": {
            "average_precision": float(average_precision_score(y, proba)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "brier_score": float(brier_score_loss(y, proba)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        },
        "rural_urban_code_subgroups": groups,
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "limitations": [
            "FARS includes fatal crashes only; this is not a model of crash occurrence or fatality risk among all road incidents.",
            "The single-year chronological split is retrospective and has no external-year validation.",
            "The 0.5 classification threshold is fixed and is not a calibrated operational threshold.",
            "Coded crash records contain missing and special-value categories; subgroup rates are descriptive.",
            "Associations are not causal and must not guide individual driving or enforcement decisions.",
        ],
    }
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

