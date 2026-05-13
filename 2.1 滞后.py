import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================
# 1️⃣ 读取数据（你的滞后文件）
# =========================
df = pd.read_csv('2.1滞后 3 7 12 18 24 30.csv', encoding='UTF-8-SIG')

# =========================
# 🔧 关键修复：删除所有访问"监测时间/数据时间"的代码
# 你的数据已经有 Date 列，直接使用即可
# =========================
# 检查 Date 列是否存在
if 'Date' not in df.columns:
    raise ValueError("❌ 数据中没有'Date'列，请检查文件！")

# 直接排序，不需要再转换时间了
df = df.sort_values(['Station', 'Date'])

y = df['y'].values

# =========================
# 2️⃣ 检查1：持久性模型
# =========================
df['y_lag1'] = df.groupby('Station')['y'].shift(1)
split_idx = int(len(df) * 0.8)
y_test_persist = y[split_idx:]
y_pred_persist = df['y_lag1'].values[split_idx:]
valid = ~np.isnan(y_pred_persist)
r2_persist = r2_score(y_test_persist[valid], y_pred_persist[valid])

print("=" * 50)
print("检查1：持久性模型（基线）")
print("=" * 50)
print(f"持久性模型在测试集上的 R²: {r2_persist:.4f}")

# =========================
# 3️⃣ 检查2：单变量线性回归
# =========================
feature = 'TP_lag7'
if feature not in df.columns:
    possible = [c for c in df.columns if 'TP' in c and 'lag7' in c]
    feature = possible[0] if possible else None

if feature:
    data = df[[feature, 'y']].dropna()
    X_single = data[[feature]].values
    y_single = data['y'].values
    split_idx_s = int(len(X_single) * 0.8)
    X_train_s, X_test_s = X_single[:split_idx_s], X_single[split_idx_s:]
    y_train_s, y_test_s = y_single[:split_idx_s], y_single[split_idx_s:]
    lr = LinearRegression()
    lr.fit(X_train_s, y_train_s)
    y_pred_s = lr.predict(X_test_s)
    r2_single = r2_score(y_test_s, y_pred_s)

    print("\n" + "=" * 50)
    print(f"检查2：单变量线性回归 ({feature} → y)")
    print("=" * 50)
    print(f"单变量模型在测试集上的 R²: {r2_single:.4f}")
else:
    print("\n❌ 未找到 TP_lag7 列，请检查列名。")