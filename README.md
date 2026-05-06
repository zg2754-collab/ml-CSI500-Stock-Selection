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
### 2. 数据获取 (Data Acquisition)
Fetch the full history of the CSI500 universe:
# Initial full download
python download_data.py --start 20200101
