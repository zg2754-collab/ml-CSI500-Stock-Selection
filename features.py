"""
Feature engineering for the CSI500 stock-selection baseline.
Final Robust Version: Dual Industry Signals (Raw Alpha + Rank Alpha).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# 🌟 双保险特征：既给模型看原始超额收益，也给它看排名
FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", 
    "vol_20d", "volume_z_20d", "turnover_ma_20d",
    "close_over_ma20", "rsi_14",
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank", 
    "volume_rank", "bias_rank", "turnover_rank",
    "price_pos_20d", "pv_divergence",
    "pv_corr_10d", "skew_20d",
    "ind_ret_norm", "ind_ret_rank"  # <--- 核心：数值+排名双信号
]

TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5

def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    close = df["close"]
    volume = df["volume"].astype(float)
    ret = close.pct_change(1)

    df["ret_1d"] = ret
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    
    df["pv_corr_10d"] = ret.rolling(10).corr(volume.pct_change(1))
    df["skew_20d"] = ret.rolling(20).skew()
    df["bias_5d"] = close / close.rolling(5).mean() - 1.0 
    df["vol_20d"] = ret.rolling(20).std()
    
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (volume - vol_mean) / vol_std
    df["pv_divergence"] = ret * (volume / (vol_mean + 1e-9))

    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan

    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0

    rolling_min = df['low'].rolling(20).min()
    rolling_max = df['high'].rolling(20).max()
    df['price_pos_20d'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-9)

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df

def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    # 1. 基础全市场排名
    rank_mapping = {
        "ret_5d": "ret_5d_rank",
        "ret_20d": "ret_20d_rank",
        "vol_20d": "vol_20d_rank",
        "volume_z_20d": "volume_rank",
        "turnover_ma_20d": "turnover_rank",
        "bias_5d": "bias_rank"
    }
    for base, rank_name in rank_mapping.items():
        if base in panel.columns:
            panel[rank_name] = panel.groupby("date")[base].rank(method="average", pct=True).fillna(0.5)

    # 2. 🌟 稳健行业信号处理
    if "industry_name" in panel.columns:
        # 计算行业中性化收益 (原始数值 - 供 L2 优化)
        ind_avg_ret = panel.groupby(["date", "industry_name"])["ret_5d"].transform("mean")
        panel["ind_ret_norm"] = panel["ret_5d"] - ind_avg_ret
        
        # 计算该中性化收益的排名 (供 Rank IC 优化)
        panel["ind_ret_rank"] = panel.groupby("date")["ind_ret_norm"].rank(method="average", pct=True)
        
        # 填充缺失
        panel["ind_ret_norm"] = panel["ind_ret_norm"].fillna(0.0)
        panel["ind_ret_rank"] = panel["ind_ret_rank"].fillna(0.5)
    else:
        panel["ind_ret_norm"] = 0.0
        panel["ind_ret_rank"] = 0.5
            
    return panel

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    
    industry_path = Path(__file__).parent / "data" / "industry_mapping.csv"
    if industry_path.exists():
        industry_df = pd.read_csv(industry_path)
        industry_df['stock_code'] = industry_df['stock_code'].astype(str).str.zfill(6)
        prices = prices.merge(industry_df[['stock_code', 'industry_name']], on='stock_code', how='left')
        prices['industry_name'] = prices['industry_name'].fillna('其他行业')
    
    all_stocks = []
    for code, group in prices.groupby("stock_code", sort=False):
        res = _per_stock_features(group)
        res["stock_code"] = code 
        all_stocks.append(res)
    
    panel = pd.concat(all_stocks, ignore_index=True)
    panel = _cross_sectional_ranks(panel)
    return panel

def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    return panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of or panel["date"].max())
    return panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
