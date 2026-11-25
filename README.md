# Regularization: Selection, Shrinkage and Valid Intervals

**ECON 5200 · Causal Machine Learning & Applied Analytics · Lab 16**

## Objective

I tested what happens to a treatment effect's interval when LASSO chooses the controls from the same data, repaired it with double selection, and measured how far LASSO shrinks the coefficients it keeps.

## Methodology

- Simulated 200 rows with a true treatment effect of 1.0, five confounders and 100 candidate controls, then ran the common procedure: LASSO picks the controls, and OLS on the survivors reports the interval.
- Ran that procedure on 200 new samples and counted how often its 95% interval contained the truth, comparing its reported standard errors with the actual spread of the estimates.
- Wrote double selection (Belloni, Chernozhukov and Hansen 2014): one LASSO of the outcome on the controls, one of the treatment on the controls, and OLS on the union of both. I compared it on the same 200 samples with OLS on the five true controls.
- Fitted LASSO on a design with 60 predictors, of which only eight matter, and compared each coefficient with the true one and with post-LASSO OLS.
- Wrote the inner loop of my own 5-fold cross-validation over 31 penalties, with the scaler fitted inside every fold, on predictors where every third column was recorded in dollars, and matched `GridSearchCV`.
- Used a `sparse_tools.py` module that returns LASSO's chosen columns, never its coefficients, and a double selection estimate with its interval.

## Key Findings

**Choosing the controls on the same data broke the interval.** The analyst's 95% interval covered the truth in 82.5% of 200 samples. Its reported standard error averaged 0.0693, while the estimates actually spread with a standard deviation of 0.0906. The mean estimate was 1.0400, above the true 1.0, even though LASSO kept all five confounders in every sample.

**Double selection helped but did not fully fix it.** Its coverage was 88.5%, with 26.5 controls kept on average, against 95.0% for OLS on the five true controls. Its mean estimate of 0.9564 sat below the truth, as the extra controls chosen on the same data absorbed part of the treatment's variation.

**LASSO coefficients are shrunk, most of all the small ones.** LASSO kept all 8 true predictors and 11 false ones. The mean shrinkage of the true coefficients was 4.6%, but the two smallest, 0.8 and −0.6, were shrunk by 15.2% and 11.1%. Post-LASSO OLS brought them back to 0.742 and −0.588.

**My own cross-validation matches GridSearchCV.** With the scaler fitted inside every fold, it chose λ = 0.0501 with a CV MSE of 1.0173, the same as `GridSearchCV` in every digit.

## Repository Contents

| File | Description |
|---|---|
| `lab-ch16-diagnostic.ipynb` | Full lab notebook with rendered output |
| `src/sparse_tools.py` | LASSO's chosen columns, a coverage study with its Monte Carlo standard error, and double selection with its 95% interval |

## Data

- All data are simulated with fixed seeds (`RANDOM_STATE = 42` for the analyst's sample, seeds 4000 to 4199 for the coverage study, and seed 7 for the shrinkage design).
