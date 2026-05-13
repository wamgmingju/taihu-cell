import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# =========================
# 1️⃣ 读取数据
# =========================
df = pd.read_csv('2.1滞后 3 7 12 18 24 30.csv', encoding='UTF-8-SIG')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Station', 'Date'])

# =========================
# 2️⃣ 准备特征（排除 EC 和 Turbidity）
# =========================
lag_cols = [col for col in df.columns if '_lag' in col]
lag_cols = [col for col in lag_cols if 'Cell' not in col and '藻密度' not in col and 'Chla' not in col]

# ✅ 新增：排除电导率(EC)和浊度(Turbidity)
exclude_params = ['EC', 'Turbidity']
lag_cols = [col for col in lag_cols if not any(ex in col for ex in exclude_params)]

print(f"排除 EC 和 Turbidity 后，特征数量: {len(lag_cols)}")
print(f"站点列表: {df['Station'].unique()}")

# =========================
# 3️⃣ 第一步：确定每个站点的 Top 5 特征
# =========================
print("\n" + "=" * 60)
print("第一步：用全特征确定每个站点的 Top 5 特征")
print("=" * 60)

top_features_by_station = {}

for station in df['Station'].unique():
    df_station = df[df['Station'] == station].sort_values('Date').copy()

    if len(df_station) < 100:
        print(f"⚠️ 站点 {station} 数据量不足 ({len(df_station)} 行)，跳过")
        continue

    X = df_station[lag_cols].values
    y = df_station['y'].values

    split_idx = int(len(X) * 0.8)
    if split_idx == 0 or len(X) - split_idx == 0:
        continue

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 用较轻的模型快速筛选
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        num_leaves=10,
        learning_rate=0.02,
        min_child_samples=30,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train_scaled, y_train)

    # 排列重要性
    perm_result = permutation_importance(model, X_test_scaled, y_test, n_repeats=5, random_state=42)
    indices = np.argsort(perm_result.importances_mean)[::-1][:5]
    top_features_by_station[station] = [lag_cols[i] for i in indices]

    print(f"  {station}: {top_features_by_station[station]}")

# =========================
# 4️⃣ 第二步：只用 Top 5 特征重新建模
# =========================
print("\n" + "=" * 60)
print("第二步：用 Top 5 特征重新建模")
print("=" * 60)

results = []
all_y_test = []
all_y_pred = []

for station, selected_cols in top_features_by_station.items():
    df_station = df[df['Station'] == station].sort_values('Date').copy()

    X = df_station[selected_cols].values
    y = df_station['y'].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        num_leaves=10,
        learning_rate=0.02,
        min_child_samples=30,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train_scaled, y_train)

    # 评估
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(
        f"站点 {station:10s} | 特征数: {len(selected_cols)} | 训练R²: {r2_train:.4f} | 测试R²: {r2_test:.4f} | 测试RMSE: {rmse_test:.4f}")

    results.append({
        'Station': station,
        'Features': ', '.join(selected_cols),
        'Train_samples': len(X_train),
        'Test_samples': len(X_test),
        'R2_train': r2_train,
        'R2_test': r2_test,
        'RMSE_test': rmse_test
    })

    all_y_test.extend(y_test)
    all_y_pred.extend(y_test_pred)

# =========================
# 5️⃣ 汇总结果
# =========================
print("\n" + "=" * 60)
print("排除 EC/Turbidity 后的 Top 5 特征建模汇总")
print("=" * 60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

overall_r2 = r2_score(all_y_test, all_y_pred)
overall_rmse = np.sqrt(mean_squared_error(all_y_test, all_y_pred))
print("\n" + "=" * 60)
print("整体评估（合并所有站点测试集）")
print("=" * 60)
print(f"整体 R²: {overall_r2:.4f}")
print(f"整体 RMSE: {overall_rmse:.4f}")

# 保存结果
results_df.to_csv('分站点建模结果_Top5特征_排除EC_Turbidity.csv', index=False, encoding='utf-8-sig')
print("\n✅ 结果已保存至 '分站点建模结果_Top5特征_排除EC_Turbidity.csv'")