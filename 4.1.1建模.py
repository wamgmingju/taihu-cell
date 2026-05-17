"""
4.1.1 建模 —— 改进版
======================
改进内容：
  1. 新增 Ridge 回归 + 线性回归基线（y_lag1 only）
  2. RF 参数优化（n_estimators=200, max_depth=5, min_samples_leaf=10）
  3. 自动读取 2.1滞后_完整版_修正.csv
  4. 逐站点 80/20 时间顺序划分（不变）
  5. 排列重要性筛选 Top5 → 三种模型对比
  6. 所有结果汇总到一张表
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

# ============================================================
# 0. 配置
# ============================================================
DATA_FILE = '2.1滞后_完整版_修正.csv'  # ← 输入文件

# RF 参数（优化后）
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_leaf': 10,
    'max_features': 0.3,
    'random_state': 42,
    'n_jobs': -1
}

# ============================================================
# 1. 读取数据
# ============================================================
print("=" * 60)
print("📂 读取数据...")
df = pd.read_csv(DATA_FILE, encoding='UTF-8-SIG')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Station', 'Date']).reset_index(drop=True)

print(f"   数据形状: {df.shape}")
print(f"   y 范围: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
print(f"   站点: {df['Station'].unique().tolist()}")

# 排除不需要的列
exclude_cols = ['Date', 'Station', 'y', 'year']
# 注意：EC 和 Turbidity 保留，在后续筛选时可选择排除

# 所有建模特征（自动识别 lag/roll/time 列）
feature_cols = [c for c in df.columns if c not in exclude_cols
                and ('_lag' in c or '_roll' in c or c in
                     ['day_of_year', 'month', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos'])]

# 识别 y_lag 特征和水质参数特征
y_lag_cols = [c for c in feature_cols if c.startswith('y_lag')]
param_lag_cols = [c for c in feature_cols if c not in y_lag_cols]

print(f"   y_lag 特征: {len(y_lag_cols)} 个")
print(f"   水质参数/时间特征: {len(param_lag_cols)} 个")
print(f"   总特征: {len(feature_cols)}")


# ============================================================
# 辅助函数
# ============================================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """训练并评估一个模型"""
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return {
        'R2_train': r2_score(y_train, y_pred_train),
        'R2_test': r2_score(y_test, y_pred_test),
        'RMSE_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'MAE_test': mean_absolute_error(y_test, y_pred_test),
        'y_pred': y_pred_test,
        'model': model
    }


def get_top5_features(model, X_train, y_train, feature_names, n=5):
    """用排列重要性筛选 Top N 特征"""
    perm = permutation_importance(
        model, X_train, y_train,
        n_repeats=5, random_state=42, n_jobs=-1
    )
    idx = np.argsort(perm.importances_mean)[::-1][:n]
    return [feature_names[i] for i in idx], perm.importances_mean[idx]


# ============================================================
# 2. 主循环：逐站点建模
# ============================================================
print("\n" + "=" * 60)
print("🔧 逐站点建模（80/20 时间顺序划分）")
print("=" * 60)

all_results = []

for station in df['Station'].unique():
    dfs = df[df['Station'] == station].sort_values('Date').copy()
    split_idx = int(len(dfs) * 0.8)

    X_all = dfs[feature_cols].values
    y_all = dfs['y'].values

    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    # 去掉含 NaN 的样本（滞后产生的）
    train_nan_mask = np.isnan(X_train).any(axis=1) | np.isnan(y_train)
    test_nan_mask = np.isnan(X_test).any(axis=1) | np.isnan(y_test)

    X_train_c = X_train[~train_nan_mask]
    y_train_c = y_train[~train_nan_mask]
    X_test_c = X_test[~test_nan_mask]
    y_test_c = y_test[~test_nan_mask]

    n_train = len(y_train_c)
    n_test = len(y_test_c)

    print(f"\n{'=' * 60}")
    print(f"📍 站点: {station}")
    print(f"   训练集: {n_train} 天 | 测试集: {n_test} 天")

    # ---------- 2a. 持久性基线（只用 y_lag1）----------
    if 'y_lag1' in feature_cols:
        idx_lag1 = feature_cols.index('y_lag1')
        y_pred_persist = X_test_c[:, idx_lag1]
        r2_persist = r2_score(y_test_c, y_pred_persist)
        rmse_persist = np.sqrt(mean_squared_error(y_test_c, y_pred_persist))
    else:
        r2_persist = np.nan
        rmse_persist = np.nan

    # ---------- 2b. 线性回归基线（只用 y_lag1）----------
    if 'y_lag1' in feature_cols:
        idx_lag1 = feature_cols.index('y_lag1')
        lr = LinearRegression()
        lr.fit(X_train_c[:, [idx_lag1]], y_train_c)
        y_pred_lr = lr.predict(X_test_c[:, [idx_lag1]])
        r2_lr = r2_score(y_test_c, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test_c, y_pred_lr))
    else:
        r2_lr = np.nan
        rmse_lr = np.nan

    # ---------- 2c. Random Forest（全特征 → Top5）----------
    # Step 1: 全特征训练 RF
    rf_full = RandomForestRegressor(**RF_PARAMS)
    res_rf_full = evaluate_model(rf_full, X_train_c, y_train_c, X_test_c, y_test_c)

    # Step 2: 排列重要性筛选 Top5
    top5_features, top5_scores = get_top5_features(
        res_rf_full['model'], X_train_c, y_train_c, feature_cols, n=5
    )
    top5_indices = [feature_cols.index(f) for f in top5_features]

    # Step 3: Top5 特征重新训练 RF
    rf_top5 = RandomForestRegressor(**RF_PARAMS)
    res_rf_top5 = evaluate_model(
        rf_top5,
        X_train_c[:, top5_indices], y_train_c,
        X_test_c[:, top5_indices], y_test_c
    )

    # ---------- 2d. Ridge 回归（Top5 特征）----------
    ridge_top5 = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    res_ridge_top5 = evaluate_model(
        ridge_top5,
        X_train_c[:, top5_indices], y_train_c,
        X_test_c[:, top5_indices], y_test_c
    )

    # ---------- 2e. 排除 EC/Turbidity 后的 Top5 ----------
    exclude_ec_turb = [c for c in feature_cols
                       if 'EC' not in c and 'Turbidity' not in c and 'turbidity' not in c]

    if set(exclude_ec_turb) != set(feature_cols):
        X_all_clean = dfs[exclude_ec_turb].values
        X_train_clean = X_all_clean[:split_idx]
        X_test_clean = X_all_clean[split_idx:]
        X_train_cc = X_train_clean[~train_nan_mask]
        X_test_cc = X_test_clean[~test_nan_mask]

        rf_clean = RandomForestRegressor(**RF_PARAMS)
        res_rf_clean = evaluate_model(rf_clean, X_train_cc, y_train_c, X_test_cc, y_test_c)

        top5_clean, top5_clean_scores = get_top5_features(
            res_rf_clean['model'], X_train_cc, y_train_c, exclude_ec_turb, n=5
        )
        top5_clean_indices = [exclude_ec_turb.index(f) for f in top5_clean]

        rf_top5_clean = RandomForestRegressor(**RF_PARAMS)
        res_rf_top5_clean = evaluate_model(
            rf_top5_clean,
            X_train_cc[:, top5_clean_indices], y_train_c,
            X_test_cc[:, top5_clean_indices], y_test_c
        )

        ridge_top5_clean = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])
        res_ridge_top5_clean = evaluate_model(
            ridge_top5_clean,
            X_train_cc[:, top5_clean_indices], y_train_c,
            X_test_cc[:, top5_clean_indices], y_test_c
        )
    else:
        top5_clean = top5_features
        res_rf_top5_clean = {'R2_test': None, 'RMSE_test': None, 'R2_train': None}
        res_ridge_top5_clean = {'R2_test': None, 'RMSE_test': None, 'R2_train': None}

    # ---------- 汇总 ----------
    row = {
        'Station': station,
        '训练天数': n_train,
        '测试天数': n_test,
        # 基线
        '持久性_R2': round(r2_persist, 4),
        '持久性_RMSE': round(rmse_persist, 4),
        '线性回归(y_lag1)_R2': round(r2_lr, 4),
        # RF 全特征
        'RF全特征_TrainR2': round(res_rf_full['R2_train'], 4),
        'RF全特征_TestR2': round(res_rf_full['R2_test'], 4),
        'RF全特征_RMSE': round(res_rf_full['RMSE_test'], 4),
        # RF Top5
        'Top5_特征': ', '.join(top5_features),
        'RF_Top5_TrainR2': round(res_rf_top5['R2_train'], 4),
        'RF_Top5_TestR2': round(res_rf_top5['R2_test'], 4),
        'RF_Top5_RMSE': round(res_rf_top5['RMSE_test'], 4),
        # Ridge Top5
        'Ridge_Top5_TrainR2': round(res_ridge_top5['R2_train'], 4),
        'Ridge_Top5_TestR2': round(res_ridge_top5['R2_test'], 4),
        'Ridge_Top5_RMSE': round(res_ridge_top5['RMSE_test'], 4),
        # RF Top5 排除 EC/Turbidity
        'Top5_排除EC_特征': ', '.join(top5_clean) if top5_clean else 'N/A',
        'RF_ExEC_TestR2': round(res_rf_top5_clean['R2_test'], 4) if res_rf_top5_clean['R2_test'] is not None else None,
        'RF_ExEC_RMSE': round(res_rf_top5_clean['RMSE_test'], 4) if res_rf_top5_clean[
                                                                        'RMSE_test'] is not None else None,
        # Ridge Top5 排除 EC/Turbidity
        'Ridge_ExEC_TestR2': round(res_ridge_top5_clean['R2_test'], 4) if res_ridge_top5_clean[
                                                                              'R2_test'] is not None else None,
        'Ridge_ExEC_RMSE': round(res_ridge_top5_clean['RMSE_test'], 4) if res_ridge_top5_clean[
                                                                              'RMSE_test'] is not None else None,
    }

    all_results.append(row)

    # 即时打印
    print(f"   📊 持久性基线          R²={r2_persist:+.4f}  RMSE={rmse_persist:.4f}")
    print(f"   📊 线性回归(y_lag1)    R²={r2_lr:+.4f}  RMSE={rmse_lr:.4f}")
    print(f"   🌲 RF 全特征           Train R²={res_rf_full['R2_train']:.4f}  Test R²={res_rf_full['R2_test']:+.4f}")
    print(
        f"   🌲 RF Top5 ({', '.join(top5_features[:3])}...)  Test R²={res_rf_top5['R2_test']:+.4f}  RMSE={res_rf_top5['RMSE_test']:.4f}")
    print(f"   🔵 Ridge Top5          Test R²={res_ridge_top5['R2_test']:+.4f}  RMSE={res_ridge_top5['RMSE_test']:.4f}")
    if res_rf_top5_clean['R2_test'] is not None:
        print(f"   🌲 RF ExEC Top5        Test R²={res_rf_top5_clean['R2_test']:+.4f}")
    if res_ridge_top5_clean['R2_test'] is not None:
        print(f"   🔵 Ridge ExEC Top5     Test R²={res_ridge_top5_clean['R2_test']:+.4f}")

# ============================================================
# 3. 汇总表
# ============================================================
results_df = pd.DataFrame(all_results)

print("\n\n" + "=" * 60)
print("📊 汇总对比表")
print("=" * 60)

# 精简版对比表
compare_cols = [
    'Station', '训练天数', '测试天数',
    '持久性_R2', '线性回归(y_lag1)_R2',
    'RF_Top5_TestR2', 'Ridge_Top5_TestR2',
    'RF_ExEC_TestR2', 'Ridge_ExEC_TestR2',
    'Top5_特征'
]
compare_df = results_df[compare_cols].copy()
print(compare_df.to_string(index=False))

# 完整版
results_df.to_csv('分站点建模结果_完整对比.csv', index=False, encoding='UTF-8-SIG')
print(f"\n✅ 完整结果已保存至: 分站点建模结果_完整对比.csv")

# ============================================================
# 4. 整体评估（合并所有站点测试集）
# ============================================================
print("\n" + "=" * 60)
print("📊 整体评估（合并三站测试集）")
print("=" * 60)

for model_name in ['持久性', '线性回归', 'RF Top5', 'Ridge Top5']:
    overall_y_true = []
    overall_y_pred = []

    for i, station in enumerate(df['Station'].unique()):
        dfs = df[df['Station'] == station].sort_values('Date').copy()
        split_idx = int(len(dfs) * 0.8)

        X_all = dfs[feature_cols].values
        y_all = dfs['y'].values

        X_train_s, X_test_s = X_all[:split_idx], X_all[split_idx:]
        y_train_s, y_test_s = y_all[:split_idx], y_all[split_idx:]

        train_mask = ~(np.isnan(X_train_s).any(axis=1) | np.isnan(y_train_s))
        test_mask = ~(np.isnan(X_test_s).any(axis=1) | np.isnan(y_test_s))

        y_test_clean = y_test_s[test_mask]
        X_test_clean = X_test_s[test_mask]
        X_train_clean = X_train_s[train_mask]
        y_train_clean = y_train_s[train_mask]

        if model_name == '持久性' and 'y_lag1' in feature_cols:
            pred = X_test_clean[:, feature_cols.index('y_lag1')]
        elif model_name == '线性回归' and 'y_lag1' in feature_cols:
            idx = feature_cols.index('y_lag1')
            lr = LinearRegression()
            lr.fit(X_train_clean[:, [idx]], y_train_clean)
            pred = lr.predict(X_test_clean[:, [idx]])
        elif model_name in ['RF Top5', 'Ridge Top5']:
            row = all_results[i]
            top5_feats = row['Top5_特征'].split(', ')
            top5_idx = [feature_cols.index(f) for f in top5_feats if f in feature_cols]
            if not top5_idx:
                continue

            if 'RF' in model_name:
                m = RandomForestRegressor(**RF_PARAMS)
            else:
                m = Ridge(alpha=1.0)
            m.fit(X_train_clean[:, top5_idx], y_train_clean)
            pred = m.predict(X_test_clean[:, top5_idx])
        else:
            continue

        overall_y_true.extend(y_test_clean)
        overall_y_pred.extend(pred)

    if overall_y_true:
        r2_overall = r2_score(overall_y_true, overall_y_pred)
        rmse_overall = np.sqrt(mean_squared_error(overall_y_true, overall_y_pred))
        print(f"   {model_name:<12s}  整体 R²={r2_overall:+.4f}  整体 RMSE={rmse_overall:.4f}")

# ============================================================
# 5. 最佳模型推荐
# ============================================================
print("\n" + "=" * 60)
print("🏆 各站点最佳模型")
print("=" * 60)

for row in all_results:
    station = row['Station']
    candidates = {
        '持久性': row['持久性_R2'],
        '线性回归': row['线性回归(y_lag1)_R2'],
        'RF Top5': row['RF_Top5_TestR2'],
        'Ridge Top5': row['Ridge_Top5_TestR2'],
    }
    if row['RF_ExEC_TestR2'] is not None:
        candidates['RF ExEC Top5'] = row['RF_ExEC_TestR2']
    if row['Ridge_ExEC_TestR2'] is not None:
        candidates['Ridge ExEC Top5'] = row['Ridge_ExEC_TestR2']

    best_model = max(candidates, key=lambda k: candidates[k] if candidates[k] is not None else -999)
    best_r2 = candidates[best_model]
    print(f"   {station:<8s} → {best_model:<15s}  Test R²={best_r2:+.4f}")
"""
补充：计算 MAE 和 TimeSeriesSplit CV-R²
========================================
在已有结果基础上，补充两个指标后保存完整对比表
"""

from sklearn.model_selection import TimeSeriesSplit

print("\n" + "=" * 60)
print("📊 补充计算：MAE + TimeSeriesSplit CV-R²")
print("=" * 60)

# 读取已有汇总结果
results_df = pd.read_csv('分站点建模结果_完整对比.csv')

mae_cols = []
cv_cols = []

for i, station in enumerate(df['Station'].unique()):
    dfs = df[df['Station'] == station].sort_values('Date').copy()
    split_idx = int(len(dfs) * 0.8)

    X_all = dfs[feature_cols].values
    y_all = dfs['y'].values

    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    train_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    test_mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))

    X_train_c = X_train[train_mask]
    y_train_c = y_train[train_mask]
    X_test_c = X_test[test_mask]
    y_test_c = y_test[test_mask]

    # --- 确定该站最佳模型和 Top5 特征 ---
    row = results_df[results_df['Station'] == station].iloc[0]

    # 持久性
    idx_lag1 = feature_cols.index('y_lag1')
    persist_pred = X_test_c[:, idx_lag1]
    persist_mae = np.mean(np.abs(y_test_c - persist_pred))

    # 线性回归
    lr = LinearRegression()
    lr.fit(X_train_c[:, [idx_lag1]], y_train_c)
    lr_pred = lr.predict(X_test_c[:, [idx_lag1]])
    lr_mae = np.mean(np.abs(y_test_c - lr_pred))

    # Ridge Top5（标准化版）
    top5_str = row['Top5_特征']
    top5_list = [f.strip() for f in top5_str.split(',') if f.strip() in feature_cols]
    top5_idx = [feature_cols.index(f) for f in top5_list]

    ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    ridge.fit(X_train_c[:, top5_idx], y_train_c)
    ridge_pred = ridge.predict(X_test_c[:, top5_idx])
    ridge_mae = np.mean(np.abs(y_test_c - ridge_pred))
    ridge_rmse = np.sqrt(np.mean((y_test_c - ridge_pred) ** 2))

    # RF Top5
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train_c[:, top5_idx], y_train_c)
    rf_pred = rf.predict(X_test_c[:, top5_idx])
    rf_mae = np.mean(np.abs(y_test_c - rf_pred))

    # --- TimeSeriesSplit CV-R²（只用训练集、时间顺序） ---
    tscv = TimeSeriesSplit(n_splits=5, test_size=max(30, n_test // 3))

    cv_r2_persist = []
    cv_r2_lr = []
    cv_r2_ridge = []
    cv_r2_rf = []

    for train_cv_idx, val_cv_idx in tscv.split(X_train_c):
        X_tr, X_val = X_train_c[train_cv_idx], X_train_c[val_cv_idx]
        y_tr, y_val = y_train_c[train_cv_idx], y_train_c[val_cv_idx]

        # 持久性
        p_pred = X_val[:, idx_lag1]
        cv_r2_persist.append(r2_score(y_val, p_pred))

        # 线性回归
        lr_cv = LinearRegression()
        lr_cv.fit(X_tr[:, [idx_lag1]], y_tr)
        cv_r2_lr.append(r2_score(y_val, lr_cv.predict(X_val[:, [idx_lag1]])))

        # Ridge Top5
        ridge_cv = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        ridge_cv.fit(X_tr[:, top5_idx], y_tr)
        cv_r2_ridge.append(r2_score(y_val, ridge_cv.predict(X_val[:, top5_idx])))

        # RF Top5
        rf_cv = RandomForestRegressor(**RF_PARAMS)
        rf_cv.fit(X_tr[:, top5_idx], y_tr)
        cv_r2_rf.append(r2_score(y_val, rf_cv.predict(X_val[:, top5_idx])))

    mae_data = {
        '持久性_MAE': round(persist_mae, 4),
        '线性回归_MAE': round(lr_mae, 4),
        'Ridge_Top5_MAE': round(ridge_mae, 4),
        'RF_Top5_MAE': round(rf_mae, 4),
        '持久性_CV-R²_mean': round(np.mean(cv_r2_persist), 4),
        '持久性_CV-R²_std': round(np.std(cv_r2_persist), 4),
        '线性回归_CV-R²_mean': round(np.mean(cv_r2_lr), 4),
        '线性回归_CV-R²_std': round(np.std(cv_r2_lr), 4),
        'Ridge_Top5_CV-R²_mean': round(np.mean(cv_r2_ridge), 4),
        'Ridge_Top5_CV-R²_std': round(np.std(cv_r2_ridge), 4),
        'RF_Top5_CV-R²_mean': round(np.mean(cv_r2_rf), 4),
        'RF_Top5_CV-R²_std': round(np.std(cv_r2_rf), 4),
    }

    for col, val in mae_data.items():
        results_df.loc[results_df['Station'] == station, col] = val

    print(f"\n  {station}:")
    print(f"    持久性 MAE={persist_mae:.4f}, CV-R²={np.mean(cv_r2_persist):.4f}±{np.std(cv_r2_persist):.4f}")
    print(f"    线性回归 MAE={lr_mae:.4f}, CV-R²={np.mean(cv_r2_lr):.4f}±{np.std(cv_r2_lr):.4f}")
    print(f"    Ridge MAE={ridge_mae:.4f}, CV-R²={np.mean(cv_r2_ridge):.4f}±{np.std(cv_r2_ridge):.4f}")
    print(f"    RF    MAE={rf_mae:.4f}, CV-R²={np.mean(cv_r2_rf):.4f}±{np.std(cv_r2_rf):.4f}")

results_df.to_csv('分站点建模结果_完整对比.csv', index=False, encoding='UTF-8-SIG')
print(f"\n✅ 完整对比表（含 MAE + CV-R²）已更新保存")
print("\n✅ 完成！")