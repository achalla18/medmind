# Data audit

- Raw rows: **920** across sites `{'Cleveland': 304, 'Hungary': 293, 'VA Long Beach': 200, 'Switzerland': 123}`
- Exact duplicate rows found (excluding the `id` column): **2** (dropped before any train/test split, keeping the first occurrence)
- Natively-missing (NaN) values per model feature: `{'cp': 0, 'restecg': 2, 'slope': 309, 'thal': 486, 'sex': 0, 'fbs': 90, 'exang': 55, 'age': 0, 'trestbps': 59, 'chol': 30, 'thalach': 55, 'oldpeak': 62, 'ca': 611}`
- `chol == 0` (impossible, recoded to missing): **172** rows (18.7% of the data)
- `trestbps == 0` (impossible, recoded to missing): **1** rows
- Target class balance (0=no disease, 1=disease): `{1: 509, 0: 411}`

## Missingness is driven by site, not chance
`ca` (fluoroscopy) and `thal` (thallium stress test) are both invasive/specialized tests. Missingness by site:
  - `ca`: Cleveland: 2%, Hungary: 99%, Switzerland: 96%, VA Long Beach: 99%
  - `thal`: Cleveland: 1%, Hungary: 90%, Switzerland: 42%, VA Long Beach: 83%
  - `slope`: Cleveland: 0%, Hungary: 64%, Switzerland: 14%, VA Long Beach: 51%

Cleveland performed both tests on almost every patient; the other three sites performed them rarely or never. This is **Missing At Random conditional on site**, not Missing Completely At Random -- dropping every row with a missing `ca` or `thal` would both shrink the sample sharply and bias it toward Cleveland-only patients, which is exactly why those rows are imputed (inside the leakage-free pipeline, fit on training folds only) rather than dropped.

## Why the chol==0 finding matters
`chol == 0` affects nearly a fifth of rows with a valid cholesterol field. If left as-is, a model would learn that a cholesterol reading of exactly 0 is strongly associated with disease status -- a data-artifact correlation (which site didn't measure cholesterol), not a medical one.

## Feature retained for diagnostics only
The `dataset` (site) column is used above to explain *why* missingness looks the way it does, but is deliberately excluded from the model's input features (see `ALL_FEATURES` in `src/data.py`) -- including it would let the model learn site-specific measurement conventions instead of clinical risk.