"""
Ensemble baseline for the CSI500 stock-selection competition.

修复 & 升级：
  1. 三个模型统一使用横截面 pct rank 作为 target（原版 raw 收益会出现负 IC）
  2. 集成权重基于验证集 Rank IC² 自适应，5% 地板防止单模型主导
  3. predict_ensemble 加横截面 z-score，消除模型间量纲差异
  4. 真正的双向 embargo：post-val embargo = FORWARD_HORIZON + 2
  5. 绕过 features.training_frame 的 bug（min_date/max_date 参数被忽略）
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from features import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
)

DATA_DIR = Path(__file__).parent / "data"

VAL_DAYS = 40
EMBARGO_DAYS = 5          # pre-val embargo
MIN_STOCKS = 30
MAX_WEIGHT = 0.045
DEFAULT_TOP_K = 30


# =====================================================================
#  Utility helpers
# =====================================================================
def _daily_rank_ic(pred: np.ndarray, target: np.ndarray, dates: pd.Series) -> float:
    df = pd.DataFrame({"pred": pred, "target": target, "date": dates.values})
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 5 or g["pred"].nunique() < 2 or g["target"].nunique() < 2:
            continue
        ic, _ = spearmanr(g["pred"], g["target"])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def cs_zscore(pred: np.ndarray, dates: pd.Series) -> np.ndarray:
    s = pd.Series(pred, index=pd.Index(dates.values, name="date"))
    z = s.groupby(level=0).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return z.to_numpy()


# =====================================================================
#  Training
# =====================================================================
def train_ensemble_models(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """三个模型统一使用横截面 pct rank 作为 target。"""

    train_dates_dt = pd.to_datetime(train_df["date"])
    max_date = train_dates_dt.max()
    days_diff = (max_date - train_dates_dt).dt.days
    weights = 1.0 / (1.0 + 0.001 * days_diff)

    X_train = train_df[FEATURE_COLUMNS]
    X_val = val_df[FEATURE_COLUMNS]

    # 三个模型统一使用每日 [0, 1] 横截面 pct rank。
    # raw 5 日收益分布非平稳，会让 XGB/CAT 出现负 IC（跨期泛化失败）。
    y_train_rank = (
        train_df.groupby("date")[TARGET_COLUMN]
        .transform(lambda s: s.rank(method="average", pct=True))
        .to_numpy()
    )
    y_val_rank = (
        val_df.groupby("date")[TARGET_COLUMN]
        .transform(lambda s: s.rank(method="average", pct=True))
        .to_numpy()
    )

    # --- 1. XGBoost ---
    print("   Training XGBoost (Rank Regression)...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        tree_method="hist", early_stopping_rounds=50,
    )
    model_xgb.fit(
        X_train, y_train_rank, sample_weight=weights,
        eval_set=[(X_val, y_val_rank)], verbose=False,
    )

    # --- 2. LightGBM ---
    print("   Training LightGBM (Rank Regression)...")
    model_lgb = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        n_estimators=1500, num_leaves=63, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_lambda=2.0,
        importance_type="gain", verbosity=-1,
    )
    model_lgb.fit(
        X_train, y_train_rank, sample_weight=weights,
        eval_set=[(X_val, y_val_rank)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)],
    )

    # --- 3. CatBoost ---
    print("   Training CatBoost (Rank Regression)...")
    model_cat = CatBoostRegressor(
        iterations=1000, depth=6, learning_rate=0.02,
        l2_leaf_reg=5, loss_function="RMSE", verbose=False,
        early_stopping_rounds=50,
    )
    model_cat.fit(X_train, y_train_rank, sample_weight=weights, eval_set=(X_val, y_val_rank))

    return model_xgb, model_lgb, model_cat


# =====================================================================
#  Adaptive ensemble weights (validation Rank IC²)
# =====================================================================
def compute_ensemble_weights(val_df, model_xgb, model_lgb, model_cat,
                             feature_cols, target_col, floor=0.05):
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].to_numpy()
    dates = val_df["date"]

    pred_xgb = model_xgb.predict(X_val)
    pred_lgb = model_lgb.predict(X_val)
    pred_cat = model_cat.predict(X_val)

    ic_xgb = _daily_rank_ic(pred_xgb, y_val, dates)
    ic_lgb = _daily_rank_ic(pred_lgb, y_val, dates)
    ic_cat = _daily_rank_ic(pred_cat, y_val, dates)

    print(f"   Validation Rank IC -> LGB: {ic_lgb:.4f} | XGB: {ic_xgb:.4f} | CAT: {ic_cat:.4f}")

    raw = np.array([
        ic_lgb ** 2 if ic_lgb > 0 else 0.0,
        ic_xgb ** 2 if ic_xgb > 0 else 0.0,
        ic_cat ** 2 if ic_cat > 0 else 0.0,
    ])

    if raw.sum() <= 1e-12:
        print("   [warn] All models have non-positive IC, fallback to equal weights")
        return 1 / 3, 1 / 3, 1 / 3

    w = raw / raw.sum()
    w = np.maximum(w, floor)
    w = w / w.sum()

    w_lgb, w_xgb, w_cat = float(w[0]), float(w[1]), float(w[2])
    print(f"   Adaptive weights    -> LGB: {w_lgb:.3f} | XGB: {w_xgb:.3f} | CAT: {w_cat:.3f}")
    return w_lgb, w_xgb, w_cat


# =====================================================================
#  Prediction
# =====================================================================
def predict_ensemble(models, X, dates, w_lgb, w_xgb, w_cat):
    m_xgb, m_lgb, m_cat = models
    p_xgb = cs_zscore(m_xgb.predict(X), dates)
    p_lgb = cs_zscore(m_lgb.predict(X), dates)
    p_cat = cs_zscore(m_cat.predict(X), dates)
    return p_lgb * w_lgb + p_xgb * w_xgb + p_cat * w_cat


# =====================================================================
#  Evaluation & portfolio construction
# =====================================================================
def rank_ic(y_true, y_pred, dates):
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(top_k).copy()
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        w[~over] += excess * w[~over] / w[~over].sum()
    return w


# =====================================================================
#  Main
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()

    if not Path(args.prices).exists():
        raise FileNotFoundError(f"找不到数据文件: {args.prices}")

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)

    for col in ["pe_ratio", "pb_ratio"]:
        if col not in prices.columns:
            print(f"   Warning: {col} missing, padding with 0.")
            prices[col] = 0.0

    print(">> Building features")
    panel = build_features(prices)

    # ---------- 数据切分 ----------
    trading_dates = np.sort(panel["date"].unique())
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-1])

    RECENT_PATCH_LEN = 20
    POST_VAL_EMBARGO = FORWARD_HORIZON + 2  # = 7

    predict_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    recent_patch_end_idx = predict_idx - FORWARD_HORIZON
    recent_patch_start_idx = recent_patch_end_idx - RECENT_PATCH_LEN + 1
    recent_patch_end = pd.Timestamp(trading_dates[recent_patch_end_idx])
    recent_patch_start = pd.Timestamp(trading_dates[recent_patch_start_idx])

    val_end_idx = recent_patch_start_idx - POST_VAL_EMBARGO
    val_end = pd.Timestamp(trading_dates[val_end_idx])

    val_start_idx = val_end_idx - VAL_DAYS + 1
    val_start = pd.Timestamp(trading_dates[val_start_idx])

    hist_train_end_idx = val_start_idx - EMBARGO_DAYS - 1
    hist_train_end = pd.Timestamp(trading_dates[hist_train_end_idx])

    # ---------- 构造数据集（绕过 training_frame 的 bug）----------
    full_clean = training_frame(panel)  # 只用它做 dropna，不要传日期参数

    val_df = full_clean[
        (full_clean["date"] >= val_start) & (full_clean["date"] <= val_end)
    ].copy()

    hist_train_df = full_clean[full_clean["date"] <= hist_train_end].copy()
    recent_train_df = full_clean[
        (full_clean["date"] >= recent_patch_start)
        & (full_clean["date"] <= recent_patch_end)
    ].copy()
    train_df = pd.concat([hist_train_df, recent_train_df], ignore_index=True)

    print(f"   [Train] Historical until {hist_train_end.date()} + recent patch {recent_patch_start.date()} ~ {recent_patch_end.date()}")
    print(f"   [Valid] {val_start.date()} ~ {val_end.date()} (post-val embargo = {POST_VAL_EMBARGO} days, pre-val embargo = {EMBARGO_DAYS} days)")
    print(f"   Rows - Train: {len(train_df):,}, Val: {len(val_df):,}")

    # ---------- 训练 ----------
    print(">> Training Ensemble (XGB + LGB + CAT)")
    models = train_ensemble_models(train_df, val_df)
    model_xgb, model_lgb, model_cat = models

    print(">> Computing adaptive ensemble weights (validation Rank IC²)")
    w_lgb, w_xgb, w_cat = compute_ensemble_weights(
        val_df=val_df,
        model_xgb=model_xgb,
        model_lgb=model_lgb,
        model_cat=model_cat,
        feature_cols=FEATURE_COLUMNS,
        target_col=TARGET_COLUMN,
        floor=0.05,
    )

    val_pred = predict_ensemble(models, val_df[FEATURE_COLUMNS], val_df["date"], w_lgb, w_xgb, w_cat)
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   Ensemble validation rank IC: {ic:.4f}")

    # ---------- 组合构建 ----------
    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=as_of_ts)
    final_scores = predict_ensemble(
        models, pred_df[FEATURE_COLUMNS], pred_df["date"], w_lgb, w_xgb, w_cat,
    )
    pred_df = pred_df.assign(score=final_scores)

    weights = build_portfolio(pred_df.set_index("stock_code")["score"], top_k=args.top_k)

    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(args.out, index=False)
    print(f">> Wrote {len(out)} names to {args.out}")

    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model_xgb, max_num_features=12, ax=ax)
    plt.title("Feature Importance (XGBoost Component)")
    plt.show()


if __name__ == "__main__":
    main()
