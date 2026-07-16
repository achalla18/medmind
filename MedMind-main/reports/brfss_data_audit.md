# BRFSS data audit

- Raw rows: **253680**
- Native missing values (entire dataframe): **0** (this is a pre-cleaned survey extract -- no NaNs to impute)
- Exact duplicate rows: **23899** (9.4% of the data)
- Target class balance (0=no history, 1=heart disease/attack history): `{0: 229787, 1: 23893}` -- substantially more imbalanced (~90/10) than the UCI dataset (~55/45)
- BMI range: 12-98, with **805** respondents above BMI 60 (physiologically extreme but not impossible -- left as-is, not recoded to missing, since very high BMI is a real condition, unlike a cholesterol reading of exactly 0)

## Why duplicates are NOT dropped here (unlike the UCI dataset)
With 21 mostly-binary/small-ordinal survey questions answered by 253,680 people, many distinct respondents legitimately share an identical answer pattern purely by chance -- unlike the UCI dataset's 2 duplicate ROWS out of 920 (which were almost certainly genuine double-entries of the same patient), treating a shared answer pattern across a quarter-million survey rows as an error and dropping it would silently and systematically bias the sample toward less common response combinations. This is a deliberate, disclosed methodological difference from how duplicates were handled in the UCI dataset, not an inconsistency.