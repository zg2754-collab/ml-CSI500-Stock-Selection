CSI500 Multi-Model Ensemble Stock Selection System
This project implements a high-performance quantitative stock-selection strategy for the CSI500 index. By combining advanced Gradient Boosting Machine (GBM) algorithms with sector-neutralized alpha signals, the model achieves a robust Rank IC of ~0.0788.

1. System Architecture & Principle
The system follows a classic quantitative pipeline: Data Ingestion -> Feature Engineering -> Triple-Segment Training -> Weighted Ensemble Prediction.

The Triple-Segment Logic
To maximize real-world profitability while avoiding overfitting, the training process uses a unique chronological split:

Historical Training Set: Deep historical data (from 2020) to learn long-term market patterns.

Validation Gap (Embargo): A 20-day "buffer zone" between training and validation to prevent information leakage.

Recent Patch: Integrating the most recent 20 days of market data directly into the training set to ensure the model captures the latest "market sentiment" before making live predictions.

Weighted Ensemble
The system aggregates predictions from three state-of-the-art models with optimized weights:

LightGBM (50%): The primary driver, optimized for high-dimensional ranking tasks.

XGBoost (30%): Captures non-linear relationships and "tail" opportunities.

CatBoost (20%): Provides robustness by handling categorical features like industry classifications effectively.

2. Data Source & Timeline
Source: Data is fetched via the akshare library, pulling from Sina/EastMoney backends.

Universe: CSI500 constituents.

Current Horizon: The data is updated incrementally to the most recent trading day (currently May 7, 2026).

Target: target_5d (Relative price change over the next 5 trading days).

3. Feature Engineering Highlights
The model utilizes a blend of raw price-volume data and advanced statistical signals:

Liquidity Signals: volume_z_20d and turnover_ma_20d (Top-performing features identifying abnormal capital flow).

Sector Neutralization: ind_ret_rank - A cross-sectional rank of stock returns relative to their specific industry peers. This removes market-wide noise and focuses on "Industry Alphas".

Price Structure: skew_20d, rsi_14, and price_pos_20d (Capturing mean reversion and trend strength).

Volume-Price Divergence: pv_corr_10d (Measuring the health of price movements).

4. Key Project Strengths
High Predictive Power: Achieved a validated Rank IC of 0.0788, significantly higher than standard baselines.

Style Neutrality: Industry ranking ensures the portfolio isn't accidentally betting on a single sector.

Dynamic Adaptation: The "Recent Patch" training logic allows the model to stay relevant in fast-rotating markets like the A-share market.

Risk Control: Automatic weight capping (MAX_WEIGHT = 0.045) to prevent single-stock concentration risk.

5. Quick Start Guide
Step 1: Data Download
Download the full history of CSI500 constituents:

Bash
python download_data.py --start 20200101
Step 2: Incremental Update (Before Trading)
Always run this to pull the latest market data before generating a portfolio:

Bash
python download_data.py --update
Step 3: Train and Predict
Run the ensemble pipeline to generate the submission.csv file containing the top 30 stock picks:

Bash
python "baseline ensembled.py"
The output will include a feature importance chart and a CSV file with stock_code and weight ready for execution.
