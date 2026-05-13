import pandas as pd
import numpy as np

# =========================
# 1️⃣ 读取清洗后的数据
# =========================
df = pd.read_csv('cell_cleaned_dataset_nodup.csv', encoding='UTF-8-SIG')

# =========================
# 2️⃣ 检查重复行
# 按 Station + Time 判定重复
# =========================
duplicates = df.duplicated(subset=['Station','Time'], keep='first')
print("重复行数量:", duplicates.sum())

# =========================
# 3️⃣ 剔除重复行
# =========================
df_cleaned = df.drop_duplicates(subset=['Station','Time'], keep='first').reset_index(drop=True)

# =========================
# 4️⃣ 转换时间列为 datetime
# =========================
df_cleaned['Time'] = pd.to_datetime(df_cleaned['Time'], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['Time'])

# =========================
# 5️⃣ 按天聚合（同站点同一天取平均）
# =========================
df_cleaned['Date'] = df_cleaned['Time'].dt.date
numeric_cols = ['Cell','Temp','pH','DO','TN','TP','Turbidity','EC','CODMn','NH3N']
# 仅保留数值列参与平均
agg_df = df_cleaned.groupby(['Station','Date'])[numeric_cols].mean().reset_index()

# =========================
# 6️⃣ 按站点 + 日期排序
# =========================
agg_df = agg_df.sort_values(by=['Station','Date']).reset_index(drop=True)


# =========================
# 7️⃣ 保存聚合后的 CSV
# =========================
agg_df.to_csv('cell_cleaned_dataset_daily.csv', index=False, encoding='UTF-8-SIG')

print("✅ 数据去重 + 日聚合完成")
print("最终数据量：行数 =", agg_df.shape[0], "列数 =", agg_df.shape[1])
print("聚合后数据示例：")
print(agg_df.head())