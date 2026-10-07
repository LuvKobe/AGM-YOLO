# Dataset preparation

Dataset images are obtained from the original public sources described in the main README.

The repository provides utilities for generating compact experiment metadata:

- `golden_y_manifest.csv` from `python -m tools.audit_golden_y ...`
- `vsb_7class_seed42/split_manifest.csv`
- `vsb_7class_seed42/conversion_report.json`

Generated image and label directories are excluded from Git tracking by default.
