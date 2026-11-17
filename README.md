# Road Safety Crash Severity Lab

A narrow retrospective baseline for distinguishing **multiple-fatality crashes from single-fatality crashes within NHTSA's fatal-crash census**. It does not estimate whether a crash will happen or whether a crash will be fatal.

## Data and run

The project uses the 2023 National FARS final-release `accident.csv` table from NHTSA. FARS is a census of fatal motor vehicle crashes, so every row in this analysis already contains at least one fatality.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py
python benchmark.py
```

The fetcher downloads the official NHTSA archive, extracts only `accident.csv`, and leaves all data local. Raw data, predictions, and model artifacts are excluded from Git.

## Observed result

A predeclared calendar split trained on January–September and held out October–December 2023. The sample had 37,769 fatal crashes (27,962 train; 9,807 holdout). Multiple fatalities occurred in 6.5% of holdout crashes. A class-weighted one-hot logistic-regression baseline scored average precision 0.100 (versus a 0.065 prevalence baseline), ROC-AUC 0.634, and Brier score 0.227.

At a fixed 0.5 threshold, precision was 9.7%, recall 59.1%, and balanced accuracy 0.604: 377 true positives and 3,503 false positives among 9,807 holdout cases. This would create a substantial review burden and is not an operational threshold. Rural/urban code subgroup rates are in `reports/benchmark.json`; small groups and large error-rate differences need further investigation.

## Method and limits

- Outcome: `FATALS >= 2`, defined from the number of deaths in each fatal-crash record.
- Predictors use crash context such as time, road configuration, lighting, and weather. Crash identifiers, fatality and occupant counts, coordinates, and post-crash injury/response fields are excluded.
- The test is the last quarter of one year, not independent-year validation. The threshold is fixed rather than calibrated for a real workflow.
- The recorded codes include missing and special-value categories. Group metrics are descriptive, not fairness or safety guarantees.
- The model learns associations within already-fatal crashes. Do not use it to infer individual risk, causal effects, enforcement priorities, or expected fatality risk for all road users.

## Source

[NHTSA FARS program](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars) · [2023 National files](https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/2023/National/)

No code license file is included yet. NHTSA data terms apply to source material.

