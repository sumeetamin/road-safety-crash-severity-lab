# Road Safety Crash Severity Lab

A reusable severity classifier with class-imbalance handling, subgroup error rates, and transparent holdout reports.

> **Portfolio prototype.** Code is provided; no benchmark results are claimed until you run it on the cited data and record the output.

## Use case

A reusable severity classifier with class-imbalance handling, subgroup error rates, and transparent holdout reports. The command below reads a local data export, applies a transparent baseline, writes holdout or decision-support outputs, and saves the fitted model where applicable.

## Data and provenance

[NHTSA Fatality Analysis Reporting System (FARS)](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars). A cleaned crash-level CSV with one binary severity outcome per record. Set the outcome, positive label, and optional demographic/location grouping explicitly. Dataset files are not bundled. Respect the source terms and cite the dataset when publishing results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download/prepare the source data locally, then run:
python main.py --input data/crashes.csv --target severe_outcome --positive-label 1 --group road_type --out outputs
```

## Outputs

The `outputs/` directory contains the model-specific results (metrics and/or ranked CSV) and a serialized estimator where applicable. Outputs are reproducible with the same input, package versions, and random seed. The repository intentionally excludes downloaded data and generated artifacts.

## Evaluation and operating notes

FARS represents fatal crashes and is not representative of all road incidents. Geographic, reporting, and exposure differences limit causal or person-level interpretations.

For a production review, preserve a time-based, untouched holdout; compare against a simple business baseline; inspect calibration and subgroup error; document data lineage; and monitor drift. The included baseline is not a production service.

## License

Code: MIT. Third-party datasets remain subject to their original terms.
