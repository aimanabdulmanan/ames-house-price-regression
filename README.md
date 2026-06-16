# ames-house-price-regression
Comparing Multivariable Linear Regression and Random Forest on the Ames Housing Dataset, with a focus on model fairness, interpretability, and generalisation.

## Overview
Predicting house prices is an important component for the real estate industry. The price is affected by many features, such as house size, neighbourhood area, and year of build. Understanding how house prices change based on these features can help real estate developers to plan their house pricing competitively in the future. Previous studies have found that Multivariable Linear Regression (MLR) and Random Forest Regressor are some of the models that can be used to predict house prices through supervised learning (Eze et al., 2023). However, these two models work differently from each other. MLR assumes linear relationships, while Random Forest captures non linear patterns.

## Key Findings

**5-fold cross validation:**
| Metric | MLR | RF (Baseline) | RF (Tuned) |
|---|---|---|---|
| CV RMSE | $34,153 | **$31,748** | $33,399 |
| CV R² | 0.82 | **0.84** | 0.82 |
| CV Adjusted R² | 0.75 | **0.78** | 0.76 |

Hyperparameter tuning with `RandomizedSearchCV` (50 candidates, 250 total fits) produced a more regularised model

## FATES Considerations

This project includes an explicit discussion of responsible data science principles as applied
to the housing domain:

1. Fairness: The `Neighbourhood` feature encodes location, which may reflect historical
redlining or racial segregation in US housing markets. A model trained on this data risks
perpetuating price underestimation in historically underserved areas, even without explicit
demographic data.

2. Accountability & Transparency: MLR offers full coefficient-level interpretability: each
β directly quantifies a feature's marginal effect on `SalePrice`. RF is a black-box model;
impurity-based feature importance provides a useful but imperfect proxy. If used for loan
decisions, RF cannot easily satisfy the legal requirement to explain rejections to applicants.

3. Ethics: A biased model could be used to justify discriminatory pricing in real estate.
Practitioners should audit model outputs for systematic disparities across neighbourhood types
before deployment.

4. Safety: Both models produced large errors on fold 4, indicating poor reliability on
extreme-value properties. Deploying either model without human expert review for high-value
or unusual properties creates meaningful financial risk.

## Limitations

1. Dataset restricted to Ames, Iowa (2006–2010); findings may not generalise to other markets, time periods, or countries
2. Impurity-based RF feature importance can overstate the importance of high-cardinality features and understate categorical variables spread across dummy columns
3. One-hot encoded neighbourhood dummies in MLR are susceptible to multicollinearity and represent deviations from a baseline category only
4. `SalePrice` is not log-transformed; residuals for high-value properties remain non-normal, which violates MLR assumptions for luxury-tier houses

## References

Key literature underpinning the methodology:

1. Breiman, L. (2001) — Random Forests. *Machine Learning*, 45(1), 5–32
2. James et al. (2013) — *An Introduction to Statistical Learning*. Springer
3. Géron, A. (2019) — *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly
4. Eze et al. (2023) — A Comparative Study for Predicting House Price Based on Machine Learning. IEEE

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.





