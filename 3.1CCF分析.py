import pandas as pd
import numpy as np

# =========================
# 1️⃣ 读取数据
# =========================
df = pd.read_csv('2.1滞后0-7.csv', encoding='UTF-8-SIG')

# =========================
# 2️⃣ 参数设置
# =========================
water_cols = ['Temp','pH','DO','TN','TP','Turbidity','EC','CODMn','NH3N']
max_lag = 7

# =========================
# 3️⃣ CCF函数（标准化版）
# =========================
def compute_ccf(x, y, max_lag):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    lags = range(0, max_lag + 1)
    ccf_vals = []

    for lag in lags:
        if lag == 0:
            corr = np.corrcoef(x, y)[0, 1]
        else:
            if len(x) > lag:
                corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            else:
                corr = np.nan
        ccf_vals.append(corr)

    return np.array(ccf_vals)

# =========================
# 4️⃣ 按站点做CCF
# =========================
results = []

for station in df['Station'].unique():

    df_s = df[df['Station'] == station].sort_values('Date')

    # 确保数据足够
    if len(df_s) < 50:
        continue

    y = df_s['Cell'].values

    for param in water_cols:

        if param not in df_s.columns:
            continue

        data = df_s[[param, 'Cell']].dropna()

        if len(data) < 30:
            continue

        x = data[param].values
        y_valid = data['Cell'].values

        ccf_vals = compute_ccf(x, y_valid, max_lag)

        # 找绝对值最大相关（更科学）
        best_lag = np.nanargmax(np.abs(ccf_vals))
        best_corr = ccf_vals[best_lag]

        results.append([station, param, best_lag, best_corr])

# =========================
# 5️⃣ 汇总结果
# =========================
ccf_df = pd.DataFrame(results, columns=['Station','Variable','Best_Lag','Correlation'])

ccf_df.to_csv('ccf_station_results.csv', index=False, encoding='UTF-8-SIG')

print("✅ 每个站点CCF完成")

# =========================
# 6️⃣ 全局统计（论文重点）
# =========================
summary = ccf_df.groupby('Variable').agg({
    'Best_Lag': ['mean','median'],
    'Correlation': 'mean'
})

summary.columns = ['Lag_Mean','Lag_Median','Corr_Mean']
summary = summary.reset_index()

# 最常见滞后（众数）
mode_lag = ccf_df.groupby('Variable')['Best_Lag'].agg(lambda x: x.value_counts().index[0]).reset_index()
mode_lag.columns = ['Variable','Lag_Mode']

summary = summary.merge(mode_lag, on='Variable')

summary.to_csv('ccf_summary.csv', index=False, encoding='UTF-8-SIG')

print("✅ 全局统计完成")

# =========================
# 7️⃣ 输出Top滞后（论文可用）
# =========================
top_lags = ccf_df.sort_values(by='Correlation', ascending=False).groupby('Variable').head(5)

top_lags.to_csv('ccf_top_lags.csv', index=False, encoding='UTF-8-SIG')

print("✅ Top滞后完成")