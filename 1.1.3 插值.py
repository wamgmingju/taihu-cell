import pandas as pd
import numpy as np

# =========================
# 1️⃣ 读取清洗后的日数据
# =========================
df = pd.read_csv('cell_cleaned_dataset_daily.csv', encoding='UTF-8-SIG')

# =========================
# 2️⃣ 确认时间列是 datetime
# =========================
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# =========================
# 3️⃣ 数值列
# =========================
num_cols = ['Cell','Temp','pH','DO','TN','TP','Turbidity','EC','CODMn','NH3N']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================
# 4️⃣ 按站点进行线性插值
# =========================
df_interp = df.sort_values(['Station','Date']).groupby('Station').apply(
    lambda group: group.interpolate(method='linear')
).reset_index(drop=True)

# =========================
# 5️⃣ 查看插值结果
# =========================
print("缺失值数量（插值后）:\n", df_interp[num_cols].isna().sum())

# =========================
# 6️⃣ 保存插值后的数据
# =========================
df_interp.to_csv('cell_cleaned_dataset_daily_interp.csv', index=False, encoding='UTF-8-SIG')
print("✅ 插值完成，文件保存为 cell_cleaned_dataset_daily_interp.csv")