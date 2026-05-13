import pandas as pd

# =========================
# 1️⃣ 读取清洗后的数据
# =========================
df = pd.read_csv('cell_cleaned_dataset.csv', encoding='UTF-8-SIG')

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
# 4️⃣ 保存去重后的 CSV
# =========================
df_cleaned.to_csv('cell_cleaned_dataset_nodup.csv', index=False, encoding='UTF-8-SIG')

print("✅ 重复数据剔除完成")
print("最终数据量：行数 =", df_cleaned.shape[0], "列数 =", df_cleaned.shape[1])