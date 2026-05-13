import pandas as pd
import numpy as np

# =========================
# 1️⃣ 读取原始数据
# =========================
df = pd.read_csv('太湖新.csv', encoding='GBK')  # 根据你的文件编码

# =========================
# 2️⃣ 删除无关列，但保留断面名称
# =========================
drop_cols = ['站点情况', '叶绿素a(mg/L)', '叶绿素a（mg/L）', 'Chla', '数据时间']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# =========================
# 3️⃣ 中文列名 → 英文列名
# =========================
rename_dict = {
    '断面名称': 'Station', '站点名称': 'Station', '站点': 'Station',
    '监测时间': 'Time',
    '藻密度(cells/L)': 'Cell',
    '水温(℃)': 'Temp', '水温（℃）': 'Temp',
    'pH(无量纲)': 'pH', 'pH（无量纲）': 'pH',
    '溶解氧(mg/L)': 'DO', '溶解氧（mg/L）': 'DO',
    '总氮(mg/L)': 'TN', '总磷(mg/L)': 'TP',
    '浊度(NTU)': 'Turbidity', '电导率(μS/cm)': 'EC',
    '高锰酸盐指数(mg/L)': 'CODMn', '氨氮(mg/L)': 'NH3N'
}

new_cols = []
for col in df.columns:
    new_col = col
    for k, v in rename_dict.items():
        if k in new_col:
            new_col = new_col.replace(k, v)
    new_cols.append(new_col)
df.columns = new_cols

# =========================
# 4️⃣ 时间列处理（监测时间）
# =========================
# 自动解析各种格式
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df = df.dropna(subset=['Time'])  # 删除无法解析的行

# =========================
# 5️⃣ 数值列清理
# =========================
num_cols = ['Cell','Temp','pH','DO','TN','TP','Turbidity','EC','CODMn','NH3N']
for col in num_cols:
    if col in df.columns:
        # 去掉逗号、空格
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 删除缺水质参数行
df = df.dropna(subset=num_cols, how='all')
df = df[df['Cell'] > 0]

# =========================
# 6️⃣ 按站点 + 时间升序排序
# =========================
df = df.sort_values(by=['Station','Time']).reset_index(drop=True)

# =========================
# 7️⃣ 可选：生成日尺度 Date 列（便于滞后计算）
df['Date'] = df['Time'].dt.date

# =========================
# 8️⃣ 保存清洗后的 CSV
df.to_csv('cell_cleaned_dataset.csv', index=False, encoding='UTF-8-SIG')

print("✅ 数据清洗完成，生成 cell_cleaned_dataset.csv")
print("数据量：行数 =", df.shape[0], "列数 =", df.shape[1])
print("监测时间示例：")
print(df['Time'].head())