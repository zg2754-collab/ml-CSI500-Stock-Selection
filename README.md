# README — CSI 500 Stock Selection Pipeline

A machine-learning pipeline that ranks CSI 500 (中证500) constituents and constructs a top-30 portfolio for a 5-trading-day forward horizon. This README explains how to install dependencies and rerun the entire pipeline end-to-end.

---

## 1. Project Overview

The pipeline ingests Chinese A-share market data, builds 21 engineered features (price/volume technicals, cross-sectional ranks, industry-neutral alpha), trains a three-model ensemble (XGBoost + LightGBM + CatBoost) with a cross-sectional rank target, blends predictions with adaptive Rank-IC² weights, and outputs a 30-stock portfolio CSV. The latest run achieved a held-out validation Rank IC of 0.1213.

---

## 2. Repository Layout

```
.
├── data/                    # Created on first run; not checked in
│   ├── prices.parquet              # OHLCV data
│   ├── index.parquet               # CSI 500 benchmark
│   ├── constituents.csv            # CSI 500 member list
│   └── industry_mapping.csv        # Industry tags
├── download_data.py         # Fetches prices, index, constituents from akshare
├── get_industry.py          # Fetches industry mapping from Tushare Pro
├── features.py              # Builds 21 features and the target
├── baseline_ensembled.py    # Main training & inference pipeline
├── score_submission.py      # Out-of-sample portfolio scoring
├── submission.csv           # Final 30-stock portfolio (output)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 3. Prerequisites

- **Python**: 3.10 or 3.11 (3.12 has not been tested with all dependencies)
- **Operating system**: Windows 10/11, macOS, or Linux
- **RAM**: 8 GB minimum, 16 GB recommended (the panel after feature construction is roughly 2 million rows)
- **Disk**: ~500 MB for cached data and model artifacts
- **Internet access**: required only for the data-download step
- **Tushare Pro account**: free registration at [tushare.pro](https://tushare.pro), needed for the industry mapping step

---

## 4. Installation

### 4.1 Clone or download the repository

```bash
cd path/to/your/workspace
# place all .py files here
```

### 4.2 Create and activate a virtual environment

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.3 Install dependencies

If `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

If you are starting from a clean repository, install the dependencies directly:

```bash
pip install pandas numpy scipy pyarrow scikit-learn \
            xgboost lightgbm catboost \
            akshare tushare \
            matplotlib tqdm
```

### 4.4 Configure the Tushare token

Open `get_industry.py` and replace the placeholder token at the top of the file with your own Tushare Pro token:

```python
MY_TOKEN = 'your_tushare_pro_token_here'
```

You can find your token after logging into [tushare.pro](https://tushare.pro) under "个人中心 → 接口TOKEN".

If you do not want to embed the token in source, set an environment variable instead and modify `get_industry.py` to read `os.environ["TUSHARE_TOKEN"]`.

---

## 5. Running the Pipeline End-to-End

The pipeline runs in four sequential stages. Each stage produces an artifact that the next stage consumes, so they must be run in order on a fresh checkout. Subsequent runs can skip stages 1 and 2 unless you want to refresh the data.

### Stage 1 — Download price and constituent data

```bash
python download_data.py
```

This fetches the CSI 500 constituent list, the index benchmark, and forward-adjusted (qfq) OHLCV data for every constituent from 2020-01-01 through the latest available trading day, saving them to `data/prices.parquet`, `data/index.parquet`, and `data/constituents.csv`. Expected runtime: roughly 15 to 30 minutes for the first run, depending on network conditions and akshare's rate limits. Subsequent runs are incremental and faster.

### Stage 2 — Download industry mapping

```bash
python get_industry.py
```

This fetches the full A-share basic information from Tushare Pro, joins it onto the CSI 500 constituents, and writes `data/industry_mapping.csv`. Expected runtime: under one minute. Required only once unless industry classifications need refreshing.

### Stage 3 — Train the model and generate the portfolio

```bash
python baseline_ensembled.py
```

This is the main pipeline and performs the following steps internally:

1. Loads `data/prices.parquet` and joins industry tags from `data/industry_mapping.csv`
2. Builds 21 features via `features.build_features`
3. Splits the panel into historical train, recent train patch, and validation segments using double-sided embargoes
4. Converts the target to per-day cross-sectional percentile rank
5. Trains XGBoost, LightGBM, and CatBoost with time-decay sample weights
6. Computes adaptive ensemble weights from each model's validation Rank IC²
7. Blends predictions via cross-sectional z-score before weighted averaging
8. Constructs a top-30 portfolio with rank-based weights, capped at 4.5% per name
9. Writes the result to `submission.csv` and displays a feature-importance plot

Expected runtime: 8 to 20 minutes depending on hardware. Memory usage peaks around 4 to 6 GB.

#### Optional command-line flags

```bash
python baseline_ensembled.py \
    --prices data/prices.parquet \
    --as-of 2026-04-30 \
    --top-k 30 \
    --out submission.csv
```

- `--prices`: path to the prices parquet (defaults to `data/prices.parquet`)
- `--as-of`: prediction date in `YYYY-MM-DD` format (defaults to the latest available trading day)
- `--top-k`: number of stocks in the portfolio (default 30)
- `--out`: output filename (default `submission.csv`)

### Stage 4 — Score the submission (optional)

If you want to evaluate a generated portfolio against the realized future return:

```bash
python score_submission.py --submission submission.csv
```

This computes the portfolio's realized return over the forward window and compares it to the CSI 500 benchmark.

---

## 6. Expected Output

A successful run of stage 3 produces console output similar to:

```
>> Loading data/prices.parquet
>> Building features
   [Train] Historical until 2026-01-07 + recent patch 2026-03-30 ~ 2026-04-27
   [Valid] 2026-01-15 ~ 2026-03-19 (post-val embargo = 7 days, pre-val embargo = 5 days)
   Rows - Train: 682,808, Val: 19,912
>> Training Ensemble (XGB + LGB + CAT)
   Training XGBoost (Rank Regression)...
   Training LightGBM (Rank Regression)...
   Training CatBoost (Rank Regression)...
>> Computing adaptive ensemble weights (validation Rank IC²)
   Validation Rank IC -> LGB: 0.1167 | XGB: 0.1200 | CAT: 0.1206
   Adaptive weights    -> LGB: 0.320 | XGB: 0.338 | CAT: 0.342
   Ensemble validation rank IC: 0.1213
>> Predicting portfolio
>> Wrote 30 names to submission.csv
```

The final `submission.csv` contains two columns, `stock_code` and `weight`, with 30 rows summing to 1.0.

---

## 7. Reproducibility Notes

- All three boosters are seeded internally; results across runs are deterministic up to small floating-point variation
- Re-running stage 1 fetches incremental price updates only; stages 2, 3, and 4 produce reproducible outputs given fixed input data
- The `--as-of` flag lets you reproduce the exact pipeline as of any historical date for backtesting purposes
- The validation period and embargo lengths are constants defined at the top of `baseline_ensembled.py` (`VAL_DAYS`, `EMBARGO_DAYS`, `RECENT_PATCH_LEN`, `POST_VAL_EMBARGO`)

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'akshare'`**
You forgot to activate the virtual environment, or the dependency install failed. Re-run the install step from section 4.3.

**`requests.exceptions.ConnectionError` during `download_data.py`**
akshare's Sina backend occasionally rate-limits or temporarily refuses connections. Wait a few minutes and retry; the script resumes from the last successful stock.

**`AssertionError: token not set` during `get_industry.py`**
You did not configure your Tushare Pro token in section 4.4.

**`FileNotFoundError: data/prices.parquet` when running `baseline_ensembled.py`**
You skipped stage 1. Run `python download_data.py` first.

**`Warning: pe_ratio missing, padding with 0` and `Warning: pb_ratio missing, padding with 0`**
The downloaded prices file does not include fundamental ratios. The pipeline pads them with zero and continues; this is harmless because the model does not currently use those columns. If you wish to enable them later, extend `download_data.py` to fetch fundamentals.

**Model trains but validation Rank IC is negative**
This usually indicates either a corrupted `industry_mapping.csv` (rerun stage 2) or a price-data issue spanning trading-halt periods. Check that the date ranges printed by the pipeline match expectations.

**Out-of-memory during feature construction**
Reduce the historical window in `download_data.py` from 2020-01-01 to a more recent start date, or run on a machine with at least 16 GB of RAM.

---

## 9. Citation and Credit

Data is provided by [akshare](https://akshare.akfamily.xyz) and [Tushare Pro](https://tushare.pro). Modeling uses [XGBoost](https://xgboost.readthedocs.io), [LightGBM](https://lightgbm.readthedocs.io), and [CatBoost](https://catboost.ai). All other code in this repository was written for this project.
