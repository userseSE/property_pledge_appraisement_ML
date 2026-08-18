# Modeling Slice 3 Findings

Scope: historical second-hand asking-price regression on the canonical 267,305-row analytical snapshot. The frozen test contains 40,104 rows; all reported error slices require n ≥ 100.

## Evaluation contract

- Main split: 187,089 train / 40,112 validation / 40,104 frozen test.
- Unseen-city diagnostic: 187,208 development-train / 40,130 development-validation / 39,967 rows from 19 completely held-out cities.
- Allowed features: `city_id`, `rooms`, `halls`, `area_sqm`, `orientation_primary`, `furnishing`, `floor_level`, `total_floors`, `building_type`.
- Baselines: global training-target median; per-city training-target median with global fallback for unseen cities; Ridge with fixed alpha 10; one fixed XGBoost configuration.
- Train-only preprocessing: numeric standardization plus one-hot encoding with unknown categories ignored. Validation is used only for XGBoost early stopping; frozen test data is transformed and scored only after fitting.
- Target quantile bins are used only to assign the main split and are then discarded. Price and area buckets are used only for post-prediction error slices.
- Unit-price fields, target buckets, record identifiers, platform signals, macro fields, location-like text, and title flags are excluded from all model features.
- No target trimming or post-split outlier removal is applied; large-area, high-price, and high-room-count observations remain in their assigned splits.

## 1. Frozen-test performance relative to the two median baselines

**Finding.** XGBoost reaches MAE 35.96, RMSE 76.72, R² 0.749, and median absolute error 18.76 in CNY 10,000 units.

**Evidence.** The city training-median baseline has MAE 56.63 and RMSE 124.18; XGBoost MAE is 36.5% lower. XGBoost used all 800 configured rounds; validation did not trigger early stopping. See `figures/model_comparison.png`.

**Limitation.** This is one deterministic split and one fixed boosting configuration; it is not a multi-model benchmark or evidence about sale prices.

## 2. Absolute errors vary across asking-price and area segments

**Finding.** The highest-MAE asking-price bucket is `500_plus` with MAE 245.28 and RMSE 367.66 (median absolute percentage error 27.6%; n=1,044).

**Evidence.** The highest-MAE area bucket is `144_plus` with MAE 76.14 (n=5,251). See `figures/model_error_slices.png`.

**Limitation.** These are absolute errors on a strongly right-skewed target, so higher-priced groups mechanically permit larger currency errors.

## 3. Error remains heterogeneous across property and city slices

**Finding.** Among published room-count slices, `6` rooms has the largest MAE at 120.74 (n=256).

**Evidence.** The highest-MAE published city slice is 北京 at 179.79 (n=299).

**Limitation.** Slice rankings are descriptive and can combine target scale, sample composition, and sparse support; they do not identify causes.

## 4. Complete-city holdout robustness diagnostic

**Finding.** XGBoost unseen-city MAE is 84.51, 2.35× the random frozen-test MAE; unseen-city RMSE is 199.12 and R² is 0.176.

**Evidence.** All 19 stress-test cities are absent from preprocessing fit, model fit, and early stopping. See `figures/unseen_city_stress.png`.

**Limitation.** This is a hash-selected city holdout robustness diagnostic. It does not prove performance for arbitrary future cities or time periods.
