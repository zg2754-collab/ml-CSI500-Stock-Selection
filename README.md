# CSI 500 Machine Learning Stock Selection

A quantitative stock selection framework for CSI 500 constituents using
ensemble machine learning models and adaptive Rank-IC-based weighting.

The pipeline ranks stocks by their predicted short-term performance and
constructs a Top-30 portfolio over a 5-trading-day investment horizon.

---

## Overview

This project develops a systematic machine-learning pipeline for
cross-sectional stock selection in the Chinese A-share market.

The framework combines:

- 21 engineered quantitative features
- XGBoost, LightGBM, and CatBoost
- Cross-sectional rank prediction
- Time-decay sample weighting
- Adaptive ensemble weighting based on validation Rank IC²
- Cross-sectional prediction normalization
- Rank-based portfolio construction
- Out-of-sample quantitative evaluation

The final model ranks CSI 500 constituents and selects the top 30 stocks
for a short-term holding period.

---

## Methodology

### 1. Data

The pipeline uses historical Chinese A-share market data and CSI 500
constituent information.

The data pipeline includes:

- Adjusted OHLCV price data
- CSI 500 constituent data
- CSI 500 benchmark index data
- Industry classification information

Historical data is downloaded and cached locally to support reproducible
experiments and backtesting.

---

### 2. Feature Engineering

The model uses **21 engineered quantitative features**, including:

- Price and return-based indicators
- Volume-related indicators
- Technical signals
- Cross-sectional ranks
- Industry-neutralized features

Features are constructed on a cross-sectional basis to improve comparability
among stocks within the same trading universe.

---

### 3. Machine Learning Models

Three gradient-boosting models are trained independently:

- **XGBoost**
- **LightGBM**
- **CatBoost**

The target is transformed into a daily cross-sectional percentile rank,
allowing the models to focus on the relative ranking of stocks rather than
absolute return prediction.

---

### 4. Ensemble Model

Instead of assigning equal weights to the three models, their predictions
are combined using an adaptive weighting scheme.

For each model, validation **Rank IC** is calculated and converted into
an adaptive weight using:

$$
w_i \propto IC_i^2
$$

Before aggregation, model predictions are normalized using
cross-sectional z-scores.

This allows stronger-performing models to receive greater influence in the
final ensemble.

---

### 5. Portfolio Construction

Stocks are ranked according to the ensemble prediction.

The strategy then:

1. Ranks all eligible CSI 500 constituents
2. Selects the top 30 stocks
3. Assigns rank-based portfolio weights
4. Applies a maximum weight cap of 4.5% per stock
5. Produces the final portfolio for the next 5-trading-day horizon

---

## Results

The latest held-out validation experiment achieved:

| Metric | Result |
|---|---:|
| Stock Universe | CSI 500 |
| Prediction Horizon | 5 trading days |
| Number of Features | 21 |
| Models | XGBoost + LightGBM + CatBoost |
| Portfolio Size | Top 30 |
| Ensemble Validation Rank IC | **0.1213** |

Individual model validation Rank IC:

| Model | Rank IC | Adaptive Weight |
|---|---:|---:|
| LightGBM | 0.1167 | 0.320 |
| XGBoost | 0.1200 | 0.338 |
| CatBoost | 0.1206 | 0.342 |
| **Ensemble** | **0.1213** | **1.000** |

The ensemble slightly improves the validation Rank IC over the individual
models, demonstrating the benefit of combining complementary model
predictions.

---

## Pipeline

```text
                    Market Data
                         │
                         ▼
              Data Collection & Cleaning
                         │
                         ▼
                Feature Engineering
                  21 Quant Features
                         │
                         ▼
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
           XGBoost    LightGBM    CatBoost
              │          │          │
              └──────────┼──────────┘
                         ▼
                Validation Rank IC
                         │
                         ▼
             Adaptive Rank-IC² Weights
                         │
                         ▼
              Cross-sectional Z-score
                         │
                         ▼
                Ensemble Prediction
                         │
                         ▼
                 Stock Ranking
                         │
                         ▼
                  Top-30 Portfolio
                         │
                         ▼
              Out-of-Sample Evaluation
