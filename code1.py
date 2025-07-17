import numpy as np
import pandas as pd
import yfinance as yf
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
import json
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# **步驟 1: 從線上獲取股票數據**
def download_stock_data(tickers, start_date, end_date):
    print("正在下載數據...")
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    print("數據下載完成。")
    return data

# 定義股票代碼與時間範圍
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
start_date = "2013-01-01"
end_date = "2024-01-01"

prices = download_stock_data(tickers, start_date, end_date)

output_file = "stock_prices.csv"  # 定義輸出的CSV文件名
prices.to_csv(output_file)

# **檢查數據**
print("檢查數據缺失情況：")
print(prices.isnull().sum())
print("\n歷史價格數據:")


# **將數據分為 70% 用於優化，30% 用於回測**
train_size = int(len(prices) * 0.7)
train_prices = prices.iloc[:train_size]  # 用於優化的訓練數據
test_prices = prices.iloc[train_size:]  # 用於回測的測試數據


# **步驟 2: 計算資產的預期收益與協方差矩陣**
returns = expected_returns.mean_historical_return(train_prices)
cov_matrix = risk_models.sample_cov(train_prices)

print("------------------------------------------------------------------------------")
print("\n預期收益 (Expected Returns):")
print(returns)
print("\n協方差矩陣 (Covariance Matrix):")
print(cov_matrix)
print("------------------------------------------------------------------------------")


#使用 Markowitz 理論計算未優化的投資組合**
# 使用 EfficientFrontier 來計算 Markowitz 最優組合
ef = EfficientFrontier(returns, cov_matrix)
ef.max_sharpe()  # 最大化夏普比率
weights_markowitz = ef.clean_weights()  # Markowitz 最優組合的權重



# **步驟 3: 定義投資組合優化問題**
class PortfolioOptimizationProblem(ElementwiseProblem):
    def __init__(self, returns, cov_matrix, transaction_costs=0.001, surrogate_model=None, risk_free_rate=0.02):
        # 更新 n_obj 為 2，因為現在有兩個目標：最大化夏普比率與最小化波動率
        super().__init__(n_var=len(returns), n_obj=2, n_constr=1, xl=0.0, xu=1.0)
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.transaction_costs = transaction_costs
        self.surrogate_model = surrogate_model
        self.risk_free_rate = risk_free_rate

    def _evaluate(self, x, out, *args, **kwargs):
        # 確保權重正規化
        x = np.clip(x, 0, 1)
        x = x / np.sum(x)

        # 使用代理模型預測目標函數值（如風險和收益）
        if self.surrogate_model:
            portfolio_return, portfolio_volatility = self.surrogate_model.predict(x.reshape(1, -1))[0]
        else:
            portfolio_return = np.dot(x, self.returns)
            portfolio_volatility = np.sqrt(np.dot(x.T, np.dot(self.cov_matrix, x)))

        # 加入交易成本的影響
        transaction_cost = self.transaction_costs * np.sum(np.abs(x))

        # 計算夏普比率
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        # 定義多目標函數: 最大化夏普比率，最小化波動率
        # 這裡返回的目標數據：最大化夏普比率與最小化波動率（夏普比率取負以進行最小化）
        out["F"] = [portfolio_volatility, -sharpe_ratio]  # 夏普比率作為最小化目標

        # 加入約束條件: 權重總和必須為 1
        out["G"] = [np.sum(x) - 1]

# **步驟 4: 定義代理模型**
def create_surrogate_model(X_train, Y_train):
    # 使用 GPR 建立代理模型
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    gp.fit(X_train, Y_train)
    return gp

# 假設我們有一些初始的訓練數據 X_train 和 Y_train
X_train = np.random.rand(10, len(tickers))  # 隨機生成一些初始樣本
Y_train = np.array([[0.02, 0.15], [0.03, 0.16], [0.04, 0.14], [0.03, 0.17], [0.05, 0.18], [0.02, 0.16], [0.03, 0.15], [0.01, 0.14], [0.06, 0.17], [0.04, 0.16]])  # 隨機生成一些對應的收益和波動率

# 創建代理模型
surrogate_model = create_surrogate_model(X_train, Y_train)

# **步驟 5: 啟動優化問題並使用代理模型**
problem = PortfolioOptimizationProblem(returns=returns.values, cov_matrix=cov_matrix.values, surrogate_model=surrogate_model)

# **步驟 6: 定義 NSGA-II 多目標優化演算法**
algorithm = NSGA2(pop_size=200)

# **步驟 7: 執行優化**
print("正在執行多目標優化...")
res = minimize(problem,
               algorithm, 
               termination=('n_gen', 100),
               seed=42,
               save_history=True,
               verbose=True)
print("優化完成。")

# **步驟 8: 儲存與解析優化結果，顯示每個投資組合的股票比率**
optimal_portfolios = pd.DataFrame({
    "Portfolio": [f"Portfolio {i + 1}" for i in range(len(res.X))],
    "Weights": [list(X) for X in res.X],
    "Volatility": [F[0] for F in res.F],
    "Sharpe Ratio": [-F[1] for F in res.F]  # 轉換夏普比率為正值
})

# 畫最前緣解(portfolios)
plt.scatter(optimal_portfolios["Volatility"], optimal_portfolios["Sharpe Ratio"], alpha=0.7)
plt.title("Risk vs. Sharpe Ratio (Pareto Front)")
plt.xlabel("Volatility")
plt.ylabel("Sharpe Ratio")
output_file4 = f"Risk_vs_Sharpe_Ratio.png"
plt.savefig(output_file4, format='png', dpi=300)
plt.show()
plt.close()


# 轉換權重為 JSON 格式，以便顯示
optimal_portfolios["Weights"] = optimal_portfolios["Weights"].apply(lambda x: json.dumps(x))

# **步驟 9:尋找最好的投資比率
max_sharpe_ratio = optimal_portfolios["Sharpe Ratio"].max()  # 找到最大夏普比率的投資組合

print("------------------------------------------------------------------------------")
# 使用 loc 查找對應的投資組合索引
best_portfolio_index = optimal_portfolios.loc[optimal_portfolios["Sharpe Ratio"] == max_sharpe_ratio].index[0]
print(f"The index of the best portfolio (with max Sharpe Ratio) is: {best_portfolio_index}")
print(optimal_portfolios.loc[best_portfolio_index])

print("------------------------------------------------------------------------------")

# 解析最佳投資組合的權重
best_weights = json.loads(optimal_portfolios.loc[best_portfolio_index, "Weights"])

# 使用條形圖顯示最優投資組合的權重分佈
plt.figure(figsize=(10, 6))
plt.bar(tickers, best_weights, color='skyblue')
plt.title(f"Weight Distribution for Best Portfolio (Max Sharpe Ratio)")
plt.xlabel("Assets")
plt.ylabel("Weights")
plt.xticks(rotation=45)
plt.tight_layout()
output_file5 = f"Weight_Distribution_for_Best_Portfolio_Max_Sharpe_Ratio.png"
plt.savefig(output_file5, format='png', dpi=300)
plt.show()
plt.close()




# **步驟 10: 將每個投資組合的權重、波動率、夏普比率保存到 CSV**
optimal_portfolios["Weights"] = optimal_portfolios["Weights"].apply(lambda x: json.loads(x))  # 將權重解析回列表



# 顯示每個投資組合的股票比率
# 我們將每個投資組合的權重與股票名稱對應
optimal_portfolios_expanded = optimal_portfolios.copy()

# 將每個投資組合的權重拆分為單獨的列
for i, ticker in enumerate(tickers):
    optimal_portfolios_expanded[ticker] = optimal_portfolios_expanded["Weights"].apply(lambda x: x[i])

# 儲存到 CSV 文件
output_file = "optimized_portfolios_with_stock_weights.csv"
optimal_portfolios_expanded.to_csv(output_file, index=False)

print(f"優化結果已成功保存至 {output_file}。")
# print(optimal_portfolios_expanded.head())


# **回測（Backtest）**

best_reture = 0
best_ind = 0

def backtest_portfolio(prices, weights):
    """
    回測函數，計算投資組合的累積回報。
    
    prices: 股票歷史價格，DataFrame 格式，列為股票代碼，行為時間（日期）
    weights: 投資組合權重，與股票數量匹配的列表或數組
    """
    # 計算每期的日回報
    returns = prices.pct_change().dropna()  # pct_change() 計算百分比變化，dropna() 去除缺失值
    
    # 投資組合的日回報：權重加權後的每日回報
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # 計算累積回報（指數回報）
    cumulative_returns = (1 + portfolio_returns).cumprod()
    # print(type(cumulative_returns))

    return cumulative_returns



# 回測所有優化後的投資組合
all_cumulative_returns = pd.DataFrame(index=test_prices.index)  # 用來存儲所有投資組合的累積回報



for i, row in optimal_portfolios.iterrows():
    # 解析每個投資組合的權重
    weights = optimal_portfolios["Weights"][i]
    weights = np.array(weights)  # 轉換為數組
    
    # 計算回測結果
    cumulative_returns = backtest_portfolio(test_prices, weights)

    ind = i + 1
    
    # 將回測結果加入到 DataFrame 中
    all_cumulative_returns[f"Portfolio {ind}"] = cumulative_returns
    # print(cumulative_returns)

best_portfolio_index = all_cumulative_returns.iloc[-1].idxmax()  # 找到累積回報最高的列名稱
best_cumulative_return = all_cumulative_returns.iloc[-1].max()  # 找到最高的累積回報值

# 顯示結果
print("-----------------------------------------------------------------------------")
print(f"The portfolio with the highest cumulative return is: {best_portfolio_index}")
print(f"The highest cumulative return is: {best_cumulative_return}")
print("------------------------------------------------------------------------------")
# 顯示所有投資組合的回測結果
plt.figure(figsize=(10, 6))
for column in all_cumulative_returns.columns:
    plt.plot(all_cumulative_returns[column], label=column)
plt.title("Backtest Results for All Portfolios")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
output_file1 = f"chart_Backtest_Results_for_All_Portfolios.png"
plt.savefig(output_file1, format='png', dpi=300)
plt.show()
plt.close()


# 儲存回測結果到 CSV
all_cumulative_returns.to_csv("all_cumulative_returns.csv")


print("------------------------------------------------------------------------------")
# **選擇高風險高回報或低風險低回報的投資組合**
# 高風險高回報：選擇波動率較高且夏普比率較高的投資組合
high_risk_high_return = optimal_portfolios.loc[optimal_portfolios["Sharpe Ratio"].idxmax()]
print("高風險高回報的投資組合：")
print(high_risk_high_return)

# 低風險低回報：選擇波動率較低的投資組合
low_risk_low_return = optimal_portfolios.loc[optimal_portfolios["Volatility"].idxmin()]
print("低風險低回報的投資組合：")
print(low_risk_low_return)

# 計算回報和波動率的中位數
median_return = optimal_portfolios["Sharpe Ratio"].median()
median_volatility = optimal_portfolios["Volatility"].median()

# 篩選出回報和波動率在中位數範圍內的投資組合
medium_risk_medium_return = optimal_portfolios[
    (optimal_portfolios["Sharpe Ratio"] >= median_return - 0.00001) & 
    (optimal_portfolios["Sharpe Ratio"] <= median_return + 0.00001) & 
    (optimal_portfolios["Volatility"] >= median_volatility - 0.00001) & 
    (optimal_portfolios["Volatility"] <= median_volatility + 0.00001)
]

print("中度風險中度回報的投資組合：")
print(medium_risk_medium_return)
print("------------------------------------------------------------------------------")

# 使用最佳投資組合的權重進行回測
cumulative_returns1 = backtest_portfolio(test_prices, medium_risk_medium_return.iloc[0].loc["Weights"])
cumulative_returns2 = backtest_portfolio(test_prices, high_risk_high_return.loc["Weights"])
cumulative_returns3 = backtest_portfolio(test_prices, low_risk_low_return.loc["Weights"])

# **顯示回測結果**
plt.figure(figsize=(10, 6))

# 繪製三個不同投資組合的累積回報曲線
plt.plot(cumulative_returns1, label="Portfolio 1 (medium risk, medium return)")
plt.plot(cumulative_returns2, label="Portfolio 2 (high risk, high return)")
plt.plot(cumulative_returns3, label="Portfolio 3 (Low Risk, Low Return)")

plt.title("Backtest Results for Different Portfolios")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
output_file2 = f"Backtest_Results_for_Different_Portfolios.png"
plt.savefig(output_file2, format='png', dpi=300)
plt.show()
plt.close()



# 回測 Markowitz 組合
cumulative_returns_markowitz = backtest_portfolio(test_prices, list(weights_markowitz.values()))

#回測最好的

# **步驟 5: 顯示回測結果**
plt.figure(figsize=(10, 6))

# 繪製兩個組合的累積回報
plt.plot(cumulative_returns_markowitz, label="Markowitz Optimized Portfolio")
plt.plot(cumulative_returns2 , label="Optimized Portfolio (NSGA-II)")

# 添加標題和標籤
plt.title("Comparison of Backtest Results (Markowitz vs Optimized Portfolio)")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
output_file3 = f"Comparison_of_Backtest_Results_Markowitz_vs_Optimized_Portfolio.png"
plt.savefig(output_file3, format='png', dpi=300)
plt.show()
plt.close()


# **步驟 6: 儲存回測結果**
# 儲存回測結果到 CSV
comparison_df = pd.DataFrame({
    "Markowitz": cumulative_returns_markowitz,
    "Optimized": cumulative_returns2 
}, index=test_prices.index)

comparison_df.to_csv("backtest_comparison.csv")

print("回測結果已成功保存至 'backtest_comparison.csv'")
