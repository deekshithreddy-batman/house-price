# House Price Prediction — ML Assignment

**Author:** Deekshith  
**Tool:** Python (Google Colab)  
**Model:** Linear Regression (scikit-learn)

---

## Problem Statement

Predict house prices (in lakhs ₹) based on property features such as area, number of bedrooms, bathrooms, house age, and distance from the city center. The goal is to understand how each feature influences price and evaluate a Linear Regression model using proper train-test methodology.

---

## Approach

The assignment is structured across six tasks:

1. **Dataset Creation** — A custom dataset of 20 house records was manually constructed with five features: `area_sqft`, `bedrooms`, `bathrooms`, `house_age_years`, and `price_lakhs` (target). A sixth feature `distance_km` was added during feature experimentation.

2. **Data Exploration** — Standard EDA was performed: head/tail inspection, shape check, data type verification, missing value detection, and a full statistical summary (`describe()`).

3. **Data Visualization** — Three plots were created:
   - Scatter plot: House area vs. price
   - Histogram: Distribution of house prices
   - Boxplot: Price grouped by number of bedrooms

4. **Model Training** — A Linear Regression model was trained on an 80/20 train-test split (`random_state=42`) using all four original features. Actual vs. predicted values were printed for inspection.

5. **Feature Experimentation** — Two experiments were compared against the baseline:
   - Removing `house_age_years` (3-feature model)
   - Adding `distance_km` (distance from city center) as a new 5th feature

6. **Overfitting Check** — The model was trained on the full dataset (no split) and evaluated on the same data, demonstrating why train-test splitting is essential for honest performance estimation.

---

## Model Used

**Linear Regression** (`sklearn.linear_model.LinearRegression`)

A simple, interpretable regression model that fits a linear relationship between input features and the continuous target variable (house price).

| Parameter | Value |
|---|---|
| Train-test split | 80% / 20% |
| Random state | 42 |
| Features (base) | `area_sqft`, `bedrooms`, `bathrooms`, `house_age_years` |
| Target | `price_lakhs` |

---

## Metrics

| Experiment | MAE (Lakhs ₹) | R² Score |
|---|---|---|
| Base model (4 features, train-test split) | Low | High (~strong fit) |
| Without `house_age_years` (3 features) | Slightly higher | Slightly lower |
| With `distance_km` added (5 features) | Lower | Higher |
| Full data, no split (overfitting check) | Very low | ~1.0 (inflated) |

**Key finding — most important feature:** `area_sqft` has the strongest linear correlation with price. Larger houses consistently command higher prices across the dataset.

**Key finding — new feature:** Adding `distance_km` (proximity to city center) improves model performance because location is a strong real-world predictor of property value.

> Note: With only 20 samples and a 80/20 split, the test set contains just 4 houses. Exact metric values are sensitive to which samples fall in the test set.

---

## Improvements

- **Larger dataset** — 20 samples is too small for reliable generalisation. A real dataset (e.g. from Kaggle's House Prices competition or a local property portal) would produce much more trustworthy metrics.
- **More features** — Location-based features (neighbourhood, proximity to schools/transport), property type, floor number, and furnishing status are strong real-world price predictors.
- **Cross-validation** — Replacing the single 80/20 split with k-fold cross-validation would give a more stable and unbiased estimate of model performance.
- **Feature scaling** — Standardising features (e.g. `StandardScaler`) is good practice before regression, especially when features are on very different scales (sqft vs. number of rooms).
- **Correlation heatmap** — A seaborn heatmap of feature correlations would quantify relationships before modelling, helping identify redundant or highly correlated features.
- **Try other models** — Ridge Regression, Lasso, or Random Forest Regressor could be compared against Linear Regression to see if non-linear relationships are being missed.
- **Residual plot** — Plotting residuals (actual − predicted) vs. predicted values would help verify that the linear regression assumptions (homoscedasticity, no systematic bias) hold.

---

## Key Learnings

- **Linear Regression works well when relationships are genuinely linear** — On this dataset, house price increases fairly predictably with area, making it a good fit for linear modelling.
- **Feature importance matters** — Removing `house_age_years` caused a small drop in performance, while adding `distance_km` improved it, confirming that not all features contribute equally.
- **Overfitting is clearly demonstrated** — Training and evaluating on the same data produces a near-perfect R², which is misleading. The train-test split gives the honest picture.
- **Small datasets amplify evaluation noise** — With only 4 test samples, a single outlier can swing MAE and R² dramatically. This is why dataset size matters even before model choice.
- **`random_state=42` is used consistently** — This ensures reproducible splits, so experiments can be compared fairly across runs.

---

## Project Structure

```
house_price_ml_assignment.ipynb   # Main notebook with all 6 tasks
README.md                         # This file
```

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```
