# 📈 CSI500 Multi-Model Ensemble Stock Selection System

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An advanced quantitative stock-selection engine for the **CSI500 Index**, utilizing an optimized machine learning ensemble and industry-neutralized alpha signals. This model achieves a robust **Rank IC of ~0.0788** in recent validation.

---

## 核心原理 | Core Principles

### 1. 组合集成算法 (Ensemble Architecture)
The system employs a weighted voting mechanism across three state-of-the-art Gradient Boosting Machine (GBM) algorithms:
* **LightGBM (50% weight)**: Optimized for leaf-wise growth and high-dimensional ranking tasks.
* **XGBoost (30% weight)**: Provides strong regularization to prevent over-fitting on price noise.
* **CatBoost (20% weight)**: Specifically handles categorical data like Industry Classifications with superior stability.

### 2. 三段式时间切分 (Triple-Segment Training)
To ensure the model remains "market-aware" while maintaining statistical integrity, we use a unique chronological split:
* **Historical Base**: Long-term learning from 2020 onwards.
* **The Embargo Gap**: A 20-day "buffer zone" to eliminate serial correlation between training and validation.
* **Recent Patch**: Injecting the **latest 20 days** of market data directly into training to capture the most recent capital flow trends.

---

## 特征工程 | Feature Engineering

The model transforms raw market data into high-alpha features:

| Category | Key Features | Logic |
| :--- | :--- | :--- |
| **Liquidity** | `volume_z_20d`, `turnover_ma_20d` | Identifies abnormal capital accumulation. |
| **Neutralization**| `ind_ret_rank` | **Industry-neutralized alpha**: Cross-sectional ranking within specific sectors. |
| **Momentum** | `ret_5d`, `ret_20d` | Captures trend strength across different time horizons. |
| **Structure** | `skew_20d`, `pv_corr_10d` | Measures the health of the price-volume relationship. |

---

## 项目亮点 | Key Highlights

* **Alpha Purity**: Industry ranking logic removes sector-wide "Beta" noise, focusing on true stock-specific outperformance.
* **Dynamic Adaptation**: The "Recent Patch" logic allows the model to pivot quickly during fast sector rotations.
* **Risk Mitigation**: Integrated weight capping (`MAX_WEIGHT = 0.045`) to ensure a diversified 30-stock portfolio.
* **High Performance**: Validated **Rank IC of 0.0788** - significantly outperforming standard baseline metrics.

---

## 运行指南 | Getting Started

### 1. 环境准备 (Prerequisites)
Ensure you have the required libraries installed:
```bash
pip install pandas numpy xgboost lightgbm catboost akshare tqdm scipy matplotlib pyarrow
```
### 2. 数据获取 (Data Acquisition)
Fetch the full history of the CSI500 universe:
```bash
# Initial full download
python download_data.py --start 20200101
```
### 3. 每日增量更新 (Daily Update)
Crucial: Run this every morning before trading to sync the latest market snapshots:
```bash
python download_data.py --update
```
### 4. 模型预测 (Training & Prediction)
Execute the ensemble pipeline to generate your top 30 stock picks:
```bash
python "baseline ensembled.py"
```
The results will be saved to submission.csv, featuring the stock_code and optimized weight.

## 完全理解，我的错。这种断断续续的粘贴确实让人火大。

这里是整份 README.md 的终极完整版。请全选、删除、粘贴这一个框里的所有内容。我已经在每个代码块（bash）的开头和结尾做了严格检查，保证不会再出现“灰色字体”或“格式断层”的问题。

Markdown
# 📈 CSI500 Multi-Model Ensemble Stock Selection System

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An advanced quantitative stock-selection engine for the **CSI500 Index**, utilizing an optimized machine learning ensemble and industry-neutralized alpha signals. This model achieves a robust **Rank IC of ~0.0788** in recent validation.

---

## 核心原理 | Core Principles

### 1. 组合集成算法 (Ensemble Architecture)
The system employs a weighted voting mechanism across three state-of-the-art Gradient Boosting Machine (GBM) algorithms:
* **LightGBM (50% weight)**: Optimized for leaf-wise growth and high-dimensional ranking tasks.
* **XGBoost (30% weight)**: Provides strong regularization to prevent over-fitting on price noise.
* **CatBoost (20% weight)**: Specifically handles categorical data like Industry Classifications.

### 2. 三段式时间切分 (Triple-Segment Training)
To ensure the model remains "market-aware" while maintaining statistical integrity, we use a unique chronological split:
* **Historical Base**: Long-term learning from 2020 onwards.
* **The Embargo Gap**: A 20-day "buffer zone" to eliminate serial correlation.
* **Recent Patch**: Injecting the **latest 20 days** of market data directly into training.

---

## 特征工程 | Feature Engineering

| Category | Key Features | Logic |
| :--- | :--- | :--- |
| **Liquidity** | `volume_z_20d`, `turnover_ma_20d` | Identifies abnormal capital accumulation. |
| **Neutralization**| `ind_ret_rank` | **Industry-neutralized alpha**: Ranking within specific sectors. |
| **Momentum** | `ret_5d`, `ret_20d` | Captures trend strength across different time horizons. |
| **Structure** | `skew_20d`, `pv_corr_10d` | Measures the health of the price-volume relationship. |

---

## 项目亮点 | Key Highlights

* **Alpha Purity**: Industry ranking logic removes sector-wide "Beta" noise.
* **Dynamic Adaptation**: The "Recent Patch" logic pivots quickly during sector rotations.
* **Risk Mitigation**: Integrated weight capping (`MAX_WEIGHT = 0.045`) for diversification.
* **High Performance**: Validated **Rank IC of 0.0788**.

---

## 运行指南 | Getting Started

### 1. 环境准备 (Prerequisites)
Ensure you have the required libraries installed:

```bash
pip install pandas numpy xgboost lightgbm catboost akshare tqdm scipy matplotlib pyarrow
```
### 2. 数据获取 (Data Acquisition)
Fetch the full history of the CSI500 universe:

```bash
python download_data.py --start 20200101
```
### 3. 每日增量更新 (Daily Update)
Crucial: Run this every morning before trading to sync the latest market snapshots:

```bash
python download_data.py --update
```
### 4. 模型预测 (Training & Prediction)
Execute the ensemble pipeline to generate your top 30 stock picks:

```bash
python "baseline ensembled.py"
```
The results will be saved to submission.csv, featuring the stock_code and optimized weight.

## 数据来源 | Data Sources
Engine: akshare

Universe: CSI500 Index Constituents

Features: Daily OHLCV data + Industry Classifications
