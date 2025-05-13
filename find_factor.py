import pandas as pd
import numpy as np
import requests
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 下载BTC/USDT价格和成交量数据
def fetch_btcusdt_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",  # 改为日线数据
        "limit": 1000  # 每次最多获取1000根K线
    }
    
    # 获取最近1000根K线
    response = requests.get(url, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    print(f"初始数据时间范围: {pd.to_datetime(df['timestamp'].min(), unit='ms')} 到 {pd.to_datetime(df['timestamp'].max(), unit='ms')}")
    
    # 循环获取更多历史数据
    while True:
        # 获取最早的时间戳
        earliest_timestamp = df['timestamp'].min()
        params['endTime'] = earliest_timestamp
        
        # 获取更早的数据
        response = requests.get(url, params=params)
        new_data = json.loads(response.text)
        
        if not new_data:  # 如果没有更多数据，退出循环
            break
            
        new_df = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        print(f"新获取数据时间范围: {pd.to_datetime(new_df['timestamp'].min(), unit='ms')} 到 {pd.to_datetime(new_df['timestamp'].max(), unit='ms')}")
        
        # 检查是否有重复的时间戳
        if new_df['timestamp'].isin(df['timestamp']).any():
            print("检测到重复时间戳，停止获取更多数据")
            break
            
        df = pd.concat([new_df, df])
        
        # 如果已经获取了足够的数据（比如5年），可以提前退出
        if len(df) >= 5 * 365:  # 5年的日线数据
            print("已达到目标数据量，停止获取更多数据")
            break
    
    # 处理数据
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # 确保没有重复的索引
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"最终数据时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"总数据量: {len(df)} 条记录")
    
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# 计算夏普比率
def calculate_sharpe_ratio(returns, factor_returns, risk_free_rate=0.0001):
    excess_returns = factor_returns - risk_free_rate
    std_dev = np.std(excess_returns)
    if std_dev == 0 or np.isnan(std_dev) or np.isinf(std_dev):
        return np.nan
    sharpe_ratio = np.mean(excess_returns) / std_dev
    return sharpe_ratio

# 主函数
def main():
    # 获取BTC/USDT数据
    btcusdt_data = fetch_btcusdt_data()

    # 计算收益率
    returns = btcusdt_data['close'].pct_change().dropna()
    btcusdt_data['Returns_1'] = btcusdt_data['close'].pct_change(periods=1)
    
    # 1. 增强的波动率反转因子 - 使用ATR和成交量
    btcusdt_data['TR'] = np.maximum(
        btcusdt_data['high'] - btcusdt_data['low'],
        np.maximum(
            abs(btcusdt_data['high'] - btcusdt_data['close'].shift(1)),
            abs(btcusdt_data['low'] - btcusdt_data['close'].shift(1))
        )
    )
    btcusdt_data['ATR'] = btcusdt_data['TR'].rolling(window=20).mean()
    btcusdt_data['Volume_Change'] = btcusdt_data['volume'].pct_change(periods=1)
    btcusdt_data['Enhanced_Vol_Reversal'] = -btcusdt_data['Returns_1'] / btcusdt_data['ATR'] * (1 + btcusdt_data['Volume_Change'])
    
    # 2. 价格通道反转因子 - 修改计算方式
    btcusdt_data['High_20'] = btcusdt_data['high'].rolling(window=20).max()
    btcusdt_data['Low_20'] = btcusdt_data['low'].rolling(window=20).min()
    btcusdt_data['Price_Channel'] = (btcusdt_data['close'] - btcusdt_data['Low_20']) / (btcusdt_data['High_20'] - btcusdt_data['Low_20'])
    
    # 使用不同的时间窗口和计算方法
    btcusdt_data['Returns_5'] = btcusdt_data['close'].pct_change(periods=5)  # 5日收益率
    btcusdt_data['Volume_MA_10'] = btcusdt_data['volume'].rolling(window=10).mean()  # 10日成交量均线
    btcusdt_data['Channel_Reversal'] = -btcusdt_data['Returns_5'] * (1 - btcusdt_data['Price_Channel']) * (btcusdt_data['volume'] / btcusdt_data['Volume_MA_10'])
    
    # 构建因子数据
    all_factors = btcusdt_data[['Enhanced_Vol_Reversal', 'Channel_Reversal']].copy()
    scaler = StandardScaler()
    all_factors = pd.DataFrame(scaler.fit_transform(all_factors), index=all_factors.index, columns=all_factors.columns)

    # 计算相关性矩阵
    correlation_matrix = all_factors.corr()
    print("\n=== 相关性矩阵 ===")
    print(correlation_matrix)
    
    # 计算最大相关性
    correlation_matrix_no_diag = correlation_matrix.where(~np.eye(correlation_matrix.shape[0], dtype=bool))
    max_corr = correlation_matrix_no_diag.abs().max().max()
    print(f"\n=== 最大相关性 ===")
    print(f"最大相关性: {max_corr}")

    # 计算夏普比率
    sharpe_ratios = {}
    for factor in all_factors.columns:
        factor_returns = all_factors[factor].reindex(returns.index).dropna()
        if len(factor_returns) == 0:
            sharpe_ratios[factor] = np.nan
            continue
        correlation = returns.corr(factor_returns)
        factor_returns = correlation * returns
        sharpe_ratios[factor] = calculate_sharpe_ratio(returns, factor_returns)

    print("\n=== 初始夏普比率 ===")
    for factor, sharpe_ratio in sharpe_ratios.items():
        print(f"{factor}: {sharpe_ratio}")

    # 反转负的夏普比率因子
    for factor in sharpe_ratios:
        if sharpe_ratios[factor] < 0:
            all_factors.loc[:, factor] = -all_factors[factor]

    # 重新计算夏普比率
    sharpe_ratios = {}
    for factor in all_factors.columns:
        factor_returns = all_factors[factor].reindex(returns.index).dropna()
        if len(factor_returns) == 0:
            sharpe_ratios[factor] = np.nan
            continue
        correlation = returns.corr(factor_returns)
        factor_returns = correlation * returns
        sharpe_ratios[factor] = calculate_sharpe_ratio(returns, factor_returns)

    print("\n=== 最终夏普比率 ===")
    for factor, sharpe_ratio in sharpe_ratios.items():
        print(f"{factor}: {sharpe_ratio}")

    # 计算平均夏普比率
    average_sharpe_ratio = np.nanmean(list(sharpe_ratios.values()))
    print(f"\n=== 平均夏普比率 ===")
    print(f"平均夏普比率: {average_sharpe_ratio}")

    # 可视化
    plt.style.use('seaborn-v0_8')
    
    # 1. 相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Factor Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # 2. 因子收益对比图
    plt.figure(figsize=(12, 6))
    for factor in all_factors.columns:
        factor_returns = all_factors[factor].reindex(returns.index).dropna()
        correlation = returns.corr(factor_returns)
        factor_returns = correlation * returns
        plt.plot(factor_returns.cumsum(), label=factor)
    plt.title('Cumulative Returns by Factor')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig('factor_returns.png')
    plt.close()
    
    # 3. 夏普比率柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(sharpe_ratios.keys(), sharpe_ratios.values())
    plt.axhline(y=average_sharpe_ratio, color='r', linestyle='--', label='Average')
    plt.title('Sharpe Ratios by Factor')
    plt.xlabel('Factor')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig('sharpe_ratios.png')
    plt.close()

if __name__ == "__main__":
    main()
