"""
XGBoost baseline for the CSI500 stock-selection competition.
Updated: Integrated JoinQuant fundamentals and Time-Decay Sample Weighting.
"""
from __future__ import annotations
import argparse

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.stats import spearmanr

import lightgbm as lgb
from catboost import CatBoostRegressor

from features import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
)

DATA_DIR = Path(__file__).parent / "data"
# 🌟 恢复为稳健的参数设置
VAL_DAYS = 20               # 验证集保持 20 天，确保规律的时效性
EMBARGO_DAYS = 5            # 隔离带拉长到 5 天，防止信息泄露
MIN_STOCKS = 30             
MAX_WEIGHT = 0.045         
DEFAULT_TOP_K = 30          

def train_ensemble_models(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """同时训练三个顶级梯度提升模型"""
    
    # 计算时间权重 (保持之前的优化)
    train_dates = pd.to_datetime(train_df["date"])
    max_date = train_dates.max()
    days_diff = (max_date - train_dates).dt.days
    weights = 1.0 / (1.0 + 0.001 * days_diff)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN]


    

    # --- 1. XGBoost ---
    print("   Training XGBoost...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        tree_method="hist", early_stopping_rounds=50
    )
    model_xgb.fit(X_train, y_train, sample_weight=weights,
                  eval_set=[(X_val, y_val)], verbose=False)

    # --- 2. LightGBM ---
    print("   Training LightGBM...")
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000, num_leaves=63, learning_rate=0.005,
        subsample=0.8, colsample_bytree=0.8, importance_type='gain',
        verbosity=-1
    )
    model_lgb.fit(X_train, y_train, sample_weight=weights,
                  eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(stopping_rounds=70)])

    # --- 3. CatBoost ---
    print("   Training CatBoost...")
    model_cat = CatBoostRegressor(
        iterations=1000, depth=6, learning_rate=0.02,
        l2_leaf_reg=5, loss_function='RMSE', verbose=False,
        early_stopping_rounds=50
    )
    model_cat.fit(X_train, y_train, sample_weight=weights, eval_set=(X_val, y_val))

    return model_xgb, model_lgb, model_cat

def predict_ensemble(models, X):
    """集成预测：三者取平均"""
    m_xgb, m_lgb, m_cat = models
    p_xgb = m_xgb.predict(X)
    p_lgb = m_lgb.predict(X)
    p_cat = m_cat.predict(X)
    
    # 也可以根据 IC 给不同模型分配权重，目前先均分 1:1:1
    return (p_xgb + p_lgb + p_cat) / 3

def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
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
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    return w

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices_with_fundamentals.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()

    if not Path(args.prices).exists():
        raise FileNotFoundError(f"Data file not found: {args.prices}")

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    
    print(">> Building features")
    panel = build_features(prices)
    
    # 确定预测日期和训练切分点
    trading_dates = np.sort(panel["date"].unique())
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-1])
    
    as_of_idx = np.searchsorted(trading_dates, np.datetime64(as_of_ts))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    
    train_pool = training_frame(panel, max_date=train_cutoff)
    valid_train_dates = np.sort(train_pool["date"].unique())
    
    print(f"DEBUG: 可用训练日期总数 = {len(valid_train_dates)}")
    
    if len(valid_train_dates) < VAL_DAYS + EMBARGO_DAYS + 5:
        raise RuntimeError("Not enough dates to train.")
    
    val_start = pd.Timestamp(valid_train_dates[-VAL_DAYS])
    train_end = pd.Timestamp(valid_train_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    
    train_df = train_pool[train_pool["date"] <= train_end].copy()
    val_df = train_pool[train_pool["date"] >= val_start].copy()
    
    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")

    # --- 🌟 修改 1: 训练集成模型 ---
    print(">> Training Ensemble (XGB + LGB + CAT)")
    models = train_ensemble_models(train_df, val_df)

    # --- 🌟 修改 2: 评估集成模型的验证集 IC ---
    val_pred = predict_ensemble(models, val_df[FEATURE_COLUMNS])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   Ensemble validation rank IC: {ic:.4f}")

    # 预测
    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=as_of_ts)
    if pred_df.empty:
        raise RuntimeError(f"No rows available for prediction on {as_of_ts.date()}")
    
    # --- 🌟 修改 3: 使用集成逻辑进行打分 ---
    final_scores = predict_ensemble(models, pred_df[FEATURE_COLUMNS])
    pred_df = pred_df.assign(score=final_scores)
    
    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    # 输出
    out_path = Path(args.out)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} max={out['weight'].max():.4f}")

    # --- 🌟 修改 4: 特征重要性可视化 ---
    # 注意：三个模型的重要性不同，通常取 XGBoost 的作为代表，或者分别画出
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(models[0], max_num_features=10, ax=ax) # 使用 models[0] 即 XGBoost
    plt.title(f"XGBoost Component Importance (Ensemble IC: {ic:.4f})")
    plt.show()
    
if __name__ == "__main__":
    main()
