"""
Feature engineering for the CSI500 stock-selection baseline.
Optimized Version: Reverting to high-IC momentum logic with basic quality shield.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 🌟 剔除冗余，回归“动能+活跃度+质量”的核心
FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", 
    "vol_20d", "volume_z_20d", "turnover_ma_20d",
    "close_over_ma20", "rsi_14",
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank", 
    "volume_rank", "bias_rank", "turnover_rank",
    "price_pos_20d", "pv_divergence",
    "pv_corr_10d", "skew_20d",
    "roe_rank", "fundamental_score"  # 仅保留最核心的质量和成长得分
]

TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Focus on high-signal price-volume dynamics."""
    df = df.sort_values("date").copy()
    close = df["close"]
    volume = df["volume"].astype(float)
    ret = close.pct_change(1)

    # --- 1. 核心收益率与动量 ---
    df["ret_1d"] = ret
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)
    
    # --- 2. 量价结构 (0.07 IC 的功臣) ---
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

    # 价格相对位置 (20d)
    rolling_min = df['low'].rolling(20).min()
    rolling_max = df['high'].rolling(20).max()
    df['price_pos_20d'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-9)

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # --- 3. 精简财务因子 (防守层) ---
    # 只保留最有用的三个，ffill 填充
    for col in ["roe", "inc_revenue_year_on_year", "inc_net_profit_year_on_year"]:
        if col in df.columns:
            df[col] = df[col].ffill()
        else:
            df[col] = np.nan

    if "inc_revenue_year_on_year" in df.columns and "inc_net_profit_year_on_year" in df.columns:
        df["growth_score"] = (df["inc_revenue_year_on_year"] + df["inc_net_profit_year_on_year"]) / 2
    else:
        df["growth_score"] = np.nan

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """计算截面排名，确保没有语法错误"""
    
    rank_mapping = {
        "ret_5d": "ret_5d_rank",
        "ret_20d": "ret_20d_rank",
        "vol_20d": "vol_20d_rank",
        "volume_z_20d": "volume_rank",
        "turnover_ma_20d": "turnover_rank",
        "bias_5d": "bias_rank",
        "roe": "roe_rank",
        "growth_score": "fundamental_score"
    }
    
    for base, rank_name in rank_mapping.items():
        if base in panel.columns:
            panel[rank_name] = panel.groupby("date")[base].rank(method="average", pct=True)
            panel[rank_name] = panel[rank_name].fillna(0.5)
            
    return panel


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """构建特征主函数"""
    required = {"date", "stock_code", "close", "volume"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Missing columns: {required - set(prices.columns)}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    
    all_stocks = []
    for code, group in prices.groupby("stock_code", sort=False):
        res = _per_stock_features(group)
        res["stock_code"] = code 
        all_stocks.append(res)
    
    panel = pd.concat(all_stocks, ignore_index=True)
    panel = _cross_sectional_ranks(panel)
    
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
    return df
