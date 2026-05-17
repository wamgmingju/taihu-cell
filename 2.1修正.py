"""
2.1 滞后特征生成 —— 完整版（修正）
================================
修正内容：
  - 自动检测 y 是否为原始 Cell 值，若是则自动取 ln
  - 确保输出的 y 与原始建模文件（9.3~18.5）一致
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 0. 用户配置
# ============================================================
INPUT_FILE = '1.1.3插值版.csv'  # ← 你的预处理文件
OUTPUT_FILE = '2.1滞后_完整版_修正.csv'

LAG_DAYS = [1, 2, 3, 4,5, 7, 10, 14, 21, 30]
PARAMS = ['Temp', 'pH', 'DO', 'TN', 'TP', 'Turbidity', 'EC', 'CODMn', 'NH3N']
ROLLING_WINDOWS = [7, 14, 30]

# ============================================================
# 1. 读取数据
# ============================================================
print("=" * 60)
print("📂 读取数据...")
df = pd.read_csv(INPUT_FILE, encoding='UTF-8-SIG')

# --- 统一日期列 ---
date_col = None
for c in ['Date', '数据时间', '监测时间', 'date', 'time']:
    if c in df.columns:
        date_col = c
        break
if date_col is None:
    raise ValueError("❌ 找不到日期列！现有列：" + str(df.columns.tolist()))
df['Date'] = pd.to_datetime(df[date_col])
if date_col != 'Date':
    df.drop(columns=[date_col], inplace=True)

# --- 统一站点列 ---
station_col = None
for c in ['Station', '站点', 'station', 'site']:
    if c in df.columns:
        station_col = c
        break
if station_col is None:
    raise ValueError("❌ 找不到站点列！")
df['Station'] = df[station_col].astype(str).str.strip()
if station_col != 'Station':
    df.drop(columns=[station_col], inplace=True)

# --- 🔧 关键修正：智能处理目标变量 ---
# 优先级：y（已对数） > Cell（原始，需取ln） > 藻密度（原始，需取ln）
y_found = False
for candidate in ['y', 'Cell', 'cell', '藻密度', '藻细胞密度']:
    if candidate in df.columns:
        raw_values = pd.to_numeric(df[candidate], errors='coerce')

        # 判断是否已是对数尺度：对数后的值通常在 5~25 范围
        # 原始 Cell 通常 > 10000
        if raw_values.max() < 100:
            # 已经是对数尺度，直接使用
            df['y'] = raw_values
            print(f"   ✅ 检测到 '{candidate}' 已是对数尺度 (max={raw_values.max():.1f})，直接作为 y")
        else:
            # 原始尺度，取自然对数
            df['y'] = np.log(raw_values.clip(lower=1))  # clip 防止 log(0)
            print(f"   🔄 检测到 '{candidate}' 为原始值 (max={raw_values.max():.0f})，已取 ln 作为 y")

        if candidate != 'y':
            df.drop(columns=[candidate], inplace=True)
        y_found = True
        break

if not y_found:
    raise ValueError("❌ 找不到目标变量列！现有列：" + str(df.columns.tolist()))

# --- 确保数值列 ---
for col in PARAMS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"   数据形状: {df.shape}")
print(f"   站点: {df['Station'].unique().tolist()}")
print(f"   日期范围: {df['Date'].min()} ~ {df['Date'].max()}")
print(f"   y 范围: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
print(f"   y 均值: {df['y'].mean():.2f}")
print(f"   （确认：对数尺度下应为 9~19 左右）\n")

# ============================================================
# 2. 逐站点生成特征
# ============================================================
print("=" * 60)
print("🔧 生成滞后特征...")

params_available = [p for p in PARAMS if p in df.columns]
print(f"   可用参数: {params_available}")
print(f"   滞后天数: {LAG_DAYS}")

all_stations = []

for station in df['Station'].unique():
    dfs = df[df['Station'] == station].sort_values('Date').copy()

    # --- 2a. 水质参数滞后 ---
    for param in params_available:
        for lag in LAG_DAYS:
            dfs[f'{param}_lag{lag}'] = dfs[param].shift(lag)

    # --- 2b. y 滞后 ---
    for lag in LAG_DAYS:
        dfs[f'y_lag{lag}'] = dfs['y'].shift(lag)

    # --- 2c. 滚动窗口 ---
    for param in params_available:
        for window in ROLLING_WINDOWS:
            dfs[f'{param}_roll{window}_mean'] = dfs[param].rolling(
                window, min_periods=max(1, window // 2)
            ).mean()
            dfs[f'{param}_roll{window}_std'] = dfs[param].rolling(
                window, min_periods=max(1, window // 2)
            ).std()

    # --- 2d. 时间特征 ---
    dfs['day_of_year'] = dfs['Date'].dt.dayofyear
    dfs['month'] = dfs['Date'].dt.month
    dfs['year'] = dfs['Date'].dt.year
    dfs['month_sin'] = np.sin(2 * np.pi * dfs['month'] / 12)
    dfs['month_cos'] = np.cos(2 * np.pi * dfs['month'] / 12)
    dfs['doy_sin'] = np.sin(2 * np.pi * dfs['day_of_year'] / 365.25)
    dfs['doy_cos'] = np.cos(2 * np.pi * dfs['day_of_year'] / 365.25)

    all_stations.append(dfs)

df_lagged = pd.concat(all_stations, ignore_index=True)
df_lagged = df_lagged.sort_values(['Station', 'Date']).reset_index(drop=True)

print(f"   滞后后形状: {df_lagged.shape}")

# ============================================================
# 3. 保存
# ============================================================
df_lagged.to_csv(OUTPUT_FILE, index=False, encoding='UTF-8-SIG')
print(f"\n✅ 已保存: {OUTPUT_FILE}")

# ============================================================
# 4. 逐站点持久性基线
# ============================================================
print("\n" + "=" * 60)
print("📊 逐站点持久性基线（对数尺度，80/20 时间顺序）")
print("=" * 60)

results = []
for station in df_lagged['Station'].unique():
    dfs = df_lagged[df_lagged['Station'] == station].sort_values('Date').copy()

    split_idx = int(len(dfs) * 0.8)

    y_train_true = dfs['y'].values[:split_idx]
    y_train_pred = dfs['y_lag1'].values[:split_idx]
    y_test_true = dfs['y'].values[split_idx:]
    y_test_pred = dfs['y_lag1'].values[split_idx:]

    train_valid = ~np.isnan(y_train_pred)
    test_valid = ~np.isnan(y_test_pred)

    r2_train = r2_score(y_train_true[train_valid], y_train_pred[train_valid])
    r2_test = r2_score(y_test_true[test_valid], y_test_pred[test_valid])

    results.append({
        'Station': station,
        'Train_R²': round(r2_train, 4),
        'Test_R²': round(r2_test, 4),
        '训练天数': sum(train_valid),
        '测试天数': sum(test_valid),
    })
    print(f"  {station:<8s}  Train R²={r2_train:+.4f}  Test R²={r2_test:+.4f}")

results_df = pd.DataFrame(results)
print(f"\n  三站平均 Test R²: {results_df['Test_R²'].mean():.4f}")

# ============================================================
# 5. 特征清单
# ============================================================
print("\n" + "=" * 60)
print("🔍 特征清单")
print("=" * 60)

lag_cols = sorted([c for c in df_lagged.columns if '_lag' in c])
y_lag_cols = [c for c in lag_cols if c.startswith('y_lag')]
param_lag_cols = [c for c in lag_cols if not c.startswith('y_lag')]
roll_cols = sorted([c for c in df_lagged.columns if '_roll' in c])
time_cols = ['day_of_year', 'month', 'year', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos']
time_cols_exist = [c for c in time_cols if c in df_lagged.columns]

print(f"  y_lag: {y_lag_cols}")
print(f"  水质参数滞后: {len(param_lag_cols)} 个")
print(f"  滚动特征: {len(roll_cols)} 个")
print(f"  时间特征: {len(time_cols_exist)} 个")
print(f"  总计可用建模特征: {len(lag_cols) + len(roll_cols) + len(time_cols_exist)}")
print(f"  ✅ y_lag1 存在: {'y_lag1' in df_lagged.columns}")

# ============================================================
# 6. 最终确认
# ============================================================
y_min, y_max = df_lagged['y'].min(), df_lagged['y'].max()
if y_max < 100:
    print(f"\n✅ y 范围正确 ({y_min:.1f}~{y_max:.1f}) —— 对数尺度，可直接建模")
else:
    print(f"\n⚠️  y 范围异常 ({y_min:.0f}~{y_max:.0f}) —— 可能仍是原始值，请检查输入文件！")

print("\n✅ 完成！下一步：将 4.1.1建模.py 中的文件路径改为本输出文件。")