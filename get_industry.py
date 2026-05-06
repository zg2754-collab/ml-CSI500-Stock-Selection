import tushare as ts
import pandas as pd
from pathlib import Path

# ==========================================
# 1. 配置区
# ==========================================

MY_TOKEN = '640917c128a663f3038e1f95fb0abed3150f13e86ae944f93bff93ae'
DATA_DIR = Path(__file__).parent / "data"

def fetch_industry_data():
    """
    原创功能：获取全市场股票的行业映射，并专门为中证500进行匹配
    """
    print(">> 正在初始化 Tushare 接口...")
    ts.set_token(MY_TOKEN)
    pro = ts.pro_api()

    # --- 步骤 1: 获取基础行业信息 ---
    # fields 包含：股票代码, 名称, 所属行业
    print(">> 正在从 Tushare 云端抓取行业映射表...")
    try:
        # L 表示上市状态的股票
        df = pro.stock_basic(exchange='', list_status='L', fields='symbol,name,industry')
    except Exception as e:
        print(f"❌ 抓取失败，请检查积分是否足够或 Token 是否有效: {e}")
        return

    # --- 步骤 2: 数据标准化 ---
    # 统一代码格式，确保 600519 这种格式能对应上
    df = df.rename(columns={
        'symbol': 'stock_code',
        'name': 'stock_name',
        'industry': 'industry_name'
    })
    
    # 处理缺失值：有些新股可能暂时没行业，填入“其他”防止模型报错
    df['industry_name'] = df['industry_name'].fillna('其他行业')

    # --- 步骤 3: 自动匹配你现有的中证500数据 ---
    constituents_path = DATA_DIR / "constituents.csv"
    if constituents_path.exists():
        print(">> 正在与本地中证500成分股列表进行匹配...")
        cons = pd.read_csv(constituents_path)
        cons['stock_code'] = cons['stock_code'].astype(str).str.zfill(6)
        
        # 只要属于中证500的行业信息
        df = df.merge(cons[['stock_code']], on='stock_code', how='inner')
    
    # --- 步骤 4: 保存结果 ---
    DATA_DIR.mkdir(exist_ok=True)
    save_path = DATA_DIR / "industry_mapping.csv"
    df.to_csv(save_path, index=False)
    
    print(f">> 成功！行业数据已保存至: {save_path}")
    print(f">> 涵盖行业数量: {df['industry_name'].nunique()} 个")
    print(df['industry_name'].value_counts().head(5)) # 打印样本最多的前5个行业

if __name__ == "__main__":
    fetch_industry_data()