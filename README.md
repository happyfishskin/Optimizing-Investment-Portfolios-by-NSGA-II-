# 多目標投資組合優化專案

這是一個使用 Python 進行的金融投資組合優化專案。此專案利用 NSGA-II 多目標遺傳演算法 來尋找一組在**風險（波動率）與回報（夏普比率）**之間達到最佳平衡的股票投資組合。
專案同時整合了代理模型（Surrogate Model）來加速優化過程，並包含一個回測框架，用以驗證優化後投資組合在歷史數據上的實際表現。

## 主要功能

* **多目標優化**: 同時優化兩個相互衝突的目標：
    * 最小化投資組合的波動率 (Minimize Volatility)
    * 最大化夏普比率 (Maximize Sharpe Ratio)
* **遺傳演算法**: 使用 `pymoo` 函式庫中的 NSGA-II 演算法來探索大量的可能性，並找出帕累托最優解（Pareto Front）。
* **代理模型輔助**: 使用高斯過程迴歸 (Gaussian Process Regressor) 作為代理模型，以減少直接評估高成本目標函數的次數，從而加速優化。
* **歷史數據回測**: 將數據分為訓練集（70%）和測試集（30%），在訓練集上進行優化，並在未見過的測試集上進行回測，以評估策略的真實成效。
* **績效比較**: 將 NSGA-II 優化出的最佳組合與傳統的馬可維茲 (Markowitz) 最大夏普比率組合進行回測績效比較。
* **結果視覺化**: 自動產生帕累托前緣圖、最佳權重分佈圖，以及多個投資組合的回測績效比較圖。

## 環境依賴

您需要安裝以下 Python 函式庫才能執行此腳本。建議在虛擬環境中進行安裝。

```bash
pip install numpy pandas yfinance pymoo pypfopt matplotlib scikit-learn
```

## 如何執行

1.  **安裝依賴**: 確保已依照上一節的指令安裝所有必要的函式庫。
2.  **執行腳本**: 在您的終端機中，直接執行 Python 腳本。

    ```bash
    python code1.py
    ```

腳本會自動從 Yahoo Finance 下載數據、執行優化、進行回測，並在當前目錄下生成多個結果檔案。

## 腳本執行流程

1.  **數據下載**: 從 Yahoo Finance 下載指定的股票代碼 (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`) 在 `2013-01-01` 到 `2024-01-01` 的歷史股價。
2.  **數據分割**: 將數據集以 70/30 的比例分割為訓練數據（用於優化）和測試數據（用於回測）。
3.  **參數計算**: 使用訓練數據計算預期年化回報率和協方差矩陣。
4.  **優化問題定義**:
    * 定義一個以波動率和負夏普比率為目標函數的 `pymoo` 優化問題。
    * 建立一個高斯過程迴歸代理模型來預測目標函數值。
5.  **執行優化**: 啟動 NSGA-II 演算法，進行 100 代的演化計算。
6.  **結果分析**:
    * 從帕累托最優解中，找出夏普比率最高的投資組合。
    * 視覺化帕累托前緣與最佳投資組合的權重分佈。
7.  **回測驗證**:
    * 使用測試數據對所有帕累托最優解以及高/中/低風險組合進行回測。
    * 同時回測傳統的馬可維茲模型作為比較基準。
8.  **儲存結果**: 將優化後的權重、回測數據和比較圖表儲存為 CSV 和 PNG 檔案。

## 輸出檔案說明

執行腳本後，您會在目錄中看到以下檔案：

* `stock_prices.csv`: 從 Yahoo Finance 下載的原始股價數據。
* `optimized_portfolios_with_stock_weights.csv`: 所有帕累托最優解的詳細資訊，包含各股票權重、波動率和夏普比率。
* `all_cumulative_returns.csv`: 所有最優投資組合在回測期間的每日累積回報。
* `backtest_comparison.csv`: NSGA-II 最佳組合與馬可維茲組合的回測累積回報比較。
* `Risk_vs_Sharpe_Ratio.png`: 帕累托前緣圖，顯示風險與夏普比率的權衡關係。
* `Weight_Distribution_for_Best_Portfolio_Max_Sharpe_Ratio.png`: 夏普比率最高的投資組合中各資產的權重分佈圖。
* `chart_Backtest_Results_for_All_Portfolios.png`: 所有帕累托最優解的回測累積回報曲線圖。
* `Backtest_Results_for_Different_Portfolios.png`: 高、中、低風險三個代表性投資組合的回測結果圖。
* `Comparison_of_Backtest_Results_Markowitz_vs_Optimized_Portfolio.png`: NSGA-II 最佳組合與馬可維茲組合的回測績效對比圖。
