"""
XGBoost baseline for the CSI500 stock-selection competition.
Fixed: Removed strict dependency on fundamentals file and added field padding.
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

# 确保你的 features.py 文件在同级目录下
from features import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
)

DATA_DIR = Path(__file__).parent / "data"

# 🌟 稳健的参数设置
VAL_DAYS = 40               
EMBARGO_DAYS = 5            
MIN_STOCKS = 30             
MAX_WEIGHT = 0.045         
DEFAULT_TOP_K = 30          

def train_ensemble_models(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """同时训练三个顶级梯度提升模型"""
    
    # 计算时间权重：越近的数据权重越高
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
        n_estimators=1500, num_leaves=63, learning_rate=0.005,
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
    """集成预测：加权平均 (LGB: 0.5, XGB: 0.3, CAT: 0.2)"""
    m_xgb, m_lgb, m_cat = models
    
    # 定义权重
    w_lgb = 0.5
    w_xgb = 0.3
    w_cat = 0.2
    
    # 打印一下当前的加权比例（可选，方便调试）
    # print(f">> Using weights: LGB({w_lgb}), XGB({w_xgb}), CAT({w_cat})")
    
    # 计算加权预测值
    pred = (m_lgb.predict(X) * w_lgb + 
            m_xgb.predict(X) * w_xgb + 
            m_cat.predict(X) * w_cat)
    
    return pred

def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20: continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho): ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")

def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    chosen = scores.sort_values(ascending=False).head(top_k).copy()
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)
    
    # 权重截断，防止单只股票仓位过重
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any(): break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        w[~over] += excess * w[~over] / w[~over].sum()
    return w

def main():
    p = argparse.ArgumentParser()
    # 🌟 修改点 1: 默认路径改为原始价格文件
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--as-of", default=None)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()

    if not Path(args.prices).exists():
        raise FileNotFoundError(f"找不到数据文件: {args.prices}")

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    
    # 🌟 修改点 2: 自动填充缺失的特征列（骗过 features.build_features）
    for col in ['pe_ratio', 'pb_ratio']:
        if col not in prices.columns:
            print(f"   Warning: {col} missing, padding with 0.")
            prices[col] = 0.0

    print(">> Building features")
    panel = build_features(prices)
    
   
    # --- 🌟 核心修改：三段式数据切分逻辑 🌟 ---
    trading_dates = np.sort(panel["date"].unique())
    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(trading_dates[-1])
    
    # 1. 确定最近的训练补丁 (Recent Train Patch): 包含预测日之前的所有最新数据
    # 这里的 FORWARD_HORIZON (5天) 是为了避开标签泄露
    recent_train_end_idx = np.searchsorted(trading_dates, np.datetime64(as_of_ts)) - FORWARD_HORIZON
    recent_train_end = pd.Timestamp(trading_dates[recent_train_end_idx])
    
    # 2. 留出 20 天的间隙 (Gap/Embargo): 确保验证集和最近训练集没有逻辑关联
    gap_days = 20
    val_end_idx = recent_train_end_idx - gap_days
    val_end = pd.Timestamp(trading_dates[val_end_idx])
    
    # 3. 验证集 (Validation Set): 往前推 40 天
    val_start_idx = val_end_idx - VAL_DAYS
    val_start = pd.Timestamp(trading_dates[val_start_idx])
    
    # 4. 历史训练集 (Historical Train): 验证集之前的全部数据
    hist_train_end_idx = val_start_idx - EMBARGO_DAYS
    hist_train_end = pd.Timestamp(trading_dates[hist_train_end_idx])

    # --- 构造数据集 ---
    # 验证集：只用来观察 IC，不参与最终下周预测的“喂料”
    val_df = training_frame(panel, min_date=val_start, max_date=val_end)

    # 训练集：历史长波 + 最近 20 天的“新鲜肌肉”
    hist_train_df = training_frame(panel, max_date=hist_train_end)
    recent_train_df = training_frame(panel, min_date=pd.Timestamp(trading_dates[val_end_idx + 1]), max_date=recent_train_end)
    train_df = pd.concat([hist_train_df, recent_train_df])

    print(f"   [Train] Historical until {hist_train_end.date()}, plus recent patch until {recent_train_end.date()}")
    print(f"   [Valid] Gap of {gap_days} days, Validation period: {val_start.date()} to {val_end.date()}")
    print(f"   Rows - Train: {len(train_df):,}, Val: {len(val_df):,}")

    # 训练与集成
    print(">> Training Ensemble (XGB + LGB + CAT)")
    models = train_ensemble_models(train_df, val_df)

    # 验证集评估
    val_pred = predict_ensemble(models, val_df[FEATURE_COLUMNS])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   Ensemble validation rank IC: {ic:.4f}")

    # 组合构建
    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=as_of_ts)
    final_scores = predict_ensemble(models, pred_df[FEATURE_COLUMNS])
    pred_df = pred_df.assign(score=final_scores)
    
    weights = build_portfolio(pred_df.set_index("stock_code")["score"], top_k=args.top_k)

    # 保存
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(args.out, index=False)
    print(f">> Wrote {len(out)} names to {args.out}")

    # 特征重要性分析
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(models[0], max_num_features=12, ax=ax) 
    plt.title(f"Feature Importance (XGBoost Component)")
    plt.show()
    
if __name__ == "__main__":
    main()
