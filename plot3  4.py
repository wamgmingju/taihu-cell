"""
Publication-Quality Figures — FINAL
=====================================
Fixes:
  - 3(e): abbreviated feature names, wider layout, no overlap
  - 4(b): no vertical text rotation
  - All: refined academic color palette (Scientific Reports style)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI线程问题
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Academic color palette
# ============================================================
C = {
    'y_lag':   '#3A6B8C',   # muted steel blue
    'param':   '#C2853A',   # warm bronze
    'time':    '#5A8F6A',   # sage green
    'persist': '#8C8C8C',   # medium grey
    'linear':  '#D4A84B',   # golden
    'ridge':   '#4A7B9C',   # medium blue
    'rf':      '#C2655B',   # muted terracotta
    'train':   '#5B8DB8',   # soft blue
    'test':    '#D4756B',   # soft red
    'dark':    '#3D3D3D',   # off-black
    'grid':    '#E0E0E0',   # light grid
}

MODEL_ORDER = ['Persistence', 'Linear (y_lag1)', 'Ridge Top5', 'RF Top5']
MODEL_COLORS = [C['persist'], C['linear'], C['ridge'], C['rf']]

STATIONS_EN = ['Lanshanzui', 'Tuoshan', 'Xidong']
STATIONS_CN = ['兰山嘴', '拖山', '锡东水厂']

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': False,
})

# ============================================================
# Load & compute
# ============================================================
df = pd.read_csv('2.1滞后_完整版_修正.csv', encoding='UTF-8-SIG')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Station', 'Date'])

feature_cols = [c for c in df.columns
                if c not in ['Date', 'Station', 'y', 'year']
                and ('_lag' in c or '_roll' in c
                     or c in ['day_of_year','month','month_sin','month_cos','doy_sin','doy_cos'])]

RF_PARAMS = {'n_estimators':200, 'max_depth':5, 'min_samples_leaf':10,
             'max_features':0.3, 'random_state':42, 'n_jobs':-1}

results_csv = pd.read_csv('分站点建模结果_完整对比.csv')

results = {}
for station_cn, station_en in zip(STATIONS_CN, STATIONS_EN):
    row = results_csv[results_csv['Station'] == station_cn].iloc[0]
    top5_str = row['Top5_特征']
    top5_feats = [f.strip() for f in top5_str.split(',') if f.strip() in feature_cols]
    top5_idx = [feature_cols.index(f) for f in top5_feats]

    dfs = df[df['Station'] == station_cn].sort_values('Date')
    split = int(len(dfs) * 0.8)
    X = dfs[feature_cols].values; y = dfs['y'].values
    Xtr, Xte = X[:split], X[split:]; ytr, yte = y[:split], y[split:]
    tm = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
    em = ~(np.isnan(Xte).any(axis=1) | np.isnan(yte))
    Xtr_c, Xte_c = Xtr[tm], Xte[em]; ytr_c, yte_c = ytr[tm], yte[em]

    idx1 = feature_cols.index('y_lag1')
    p_r2 = r2_score(yte_c, Xte_c[:, idx1])
    p_rmse = np.sqrt(mean_squared_error(yte_c, Xte_c[:, idx1]))
    p_mae = mean_absolute_error(yte_c, Xte_c[:, idx1])

    lr = LinearRegression(); lr.fit(Xtr_c[:,[idx1]], ytr_c)
    lr_pred = lr.predict(Xte_c[:,[idx1]])
    lr_r2 = r2_score(yte_c, lr_pred); lr_rmse = np.sqrt(mean_squared_error(yte_c, lr_pred))
    lr_mae = mean_absolute_error(yte_c, lr_pred)

    ridge = Pipeline([('s',StandardScaler()),('r',Ridge(1.0))])
    ridge.fit(Xtr_c[:,top5_idx], ytr_c)
    ridge_pred = ridge.predict(Xte_c[:,top5_idx])
    ridge_r2 = r2_score(yte_c, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(yte_c, ridge_pred))
    ridge_mae = mean_absolute_error(yte_c, ridge_pred)
    ridge_tr = ridge.score(Xtr_c[:,top5_idx], ytr_c)

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(Xtr_c[:,top5_idx], ytr_c)
    rf_pred = rf.predict(Xte_c[:,top5_idx])
    rf_r2 = r2_score(yte_c, rf_pred); rf_rmse = np.sqrt(mean_squared_error(yte_c, rf_pred))
    rf_mae = mean_absolute_error(yte_c, rf_pred)

    rf_full = RandomForestRegressor(**RF_PARAMS)
    rf_full.fit(Xtr_c, ytr_c)
    gini = rf_full.feature_importances_
    rf_tr = rf_full.score(Xtr_c, ytr_c)

    # CV-R²
    tscv = TimeSeriesSplit(n_splits=5, test_size=max(30, len(Xte_c)//3))
    cv_p, cv_l, cv_ri, cv_rf = [], [], [], []
    for ti, vi in tscv.split(Xtr_c):
        xt, xv = Xtr_c[ti], Xtr_c[vi]; yt, yv = ytr_c[ti], ytr_c[vi]
        cv_p.append(r2_score(yv, xv[:, idx1]))
        lc = LinearRegression(); lc.fit(xt[:,[idx1]], yt)
        cv_l.append(r2_score(yv, lc.predict(xv[:,[idx1]])))
        rc = Pipeline([('s',StandardScaler()),('r',Ridge(1.0))]); rc.fit(xt[:,top5_idx], yt)
        cv_ri.append(r2_score(yv, rc.predict(xv[:,top5_idx])))
        rfc = RandomForestRegressor(**RF_PARAMS); rfc.fit(xt[:,top5_idx], yt)
        cv_rf.append(r2_score(yv, rfc.predict(xv[:,top5_idx])))

    y_imp = sum(gini[i] for i, f in enumerate(feature_cols) if f.startswith('y_lag'))
    p_imp = sum(gini[i] for i, f in enumerate(feature_cols)
                if not f.startswith('y_lag') and not ('month' in f or 'doy' in f))
    t_imp = sum(gini[i] for i, f in enumerate(feature_cols)
                if 'month' in f or 'doy' in f)

    results[station_en] = {
        'dfs': dfs, 'split': split,
        'Xtr': Xtr_c, 'Xte': Xte_c, 'ytr': ytr_c, 'yte': yte_c,
        'top5_feats': top5_feats, 'gini': gini,
        'cat_gini': {'y_lag': y_imp, 'param': p_imp, 'time': t_imp},
        'metrics': {
            'Persistence':     (p_r2, p_rmse, p_mae, cv_p),
            'Linear (y_lag1)': (lr_r2, lr_rmse, lr_mae, cv_l),
            'Ridge Top5':      (ridge_r2, ridge_rmse, ridge_mae, cv_ri),
            'RF Top5':         (rf_r2, rf_rmse, rf_mae, cv_rf),
        },
        'rf_pred': rf_pred, 'ridge_pred': ridge_pred,
        'rf_train_r2': rf_tr, 'ridge_train_r2': ridge_tr,
    }

# ============================================================
# Helper: abbreviate feature names for tight spaces
# ============================================================
def abbrev(feat_list):
    """y_lag1 -> y1, CODMn_lag1 -> COD1, doy_sin -> doy.sin, etc."""
    out = []
    for f in feat_list:
        f = f.replace('_lag', '').replace('_roll', '.r').replace('_mean','m').replace('_std','s')
        f = f.replace('day_of_year', 'doy').replace('month_sin', 'm.sin').replace('month_cos', 'm.cos')
        f = f.replace('doy_sin', 'd.sin').replace('doy_cos', 'd.cos')
        out.append(f)
    return out

# ============================================================
# FIGURE 1 — Time Series + ACF
# ============================================================
def figure1():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (sc, se) in enumerate(zip(STATIONS_CN, STATIONS_EN)):
        r = results[se]
        dfs_s = r['dfs']; split = r['split']
        train_end = dfs_s['Date'].iloc[split]

        ax = axes[0, i]
        ax.plot(dfs_s['Date'], np.exp(dfs_s['y']), color=C['dark'], lw=0.6, alpha=0.85)
        ax.axvline(train_end, color=C['test'], ls='--', lw=1.2, alpha=0.8)
        ax.set_yscale('log')
        ax.set_ylabel('Algal density (cells/L)')
        ax.set_title(f'{chr(97+i)}  {se}', fontweight='bold', loc='left', color=C['dark'])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        if i == 0:
            ax.legend(['Observed', 'Train/test split'], fontsize=7, loc='upper left', framealpha=0.85)

        ax = axes[1, i]
        plot_acf(dfs_s['y'].dropna(), lags=30, ax=ax, alpha=0.05, color=C['dark'], title='')
        ax.set_ylim(-0.35, 1.05)
        ax.set_xlabel('Lag (days)'); ax.set_ylabel('ACF')
        ax.set_title(f'{chr(100+i)}  {se}  ACF', fontweight='bold', loc='left', color=C['dark'])
        acf1 = dfs_s['y'].autocorr(lag=1)
        ax.annotate(f'ACF(1)={acf1:.2f}', xy=(1, acf1), xytext=(6, acf1+0.12),
                   fontsize=8, arrowprops=dict(arrowstyle='->', color=C['rf'], lw=0.8),
                   color=C['rf'], fontweight='bold')

    fig.suptitle('Figure 1. Time series and autocorrelation structure of log-transformed algal density',
                 fontweight='bold', fontsize=12, color=C['dark'], y=1.01)
    plt.tight_layout()
    fig.savefig('Figure1_FINAL.pdf')
    plt.close()
    print('Saved -> Figure1_FINAL.pdf')

# ============================================================
# FIGURE 2 — Observed vs Predicted
# ============================================================
def figure2():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, (sc, se) in enumerate(zip(STATIONS_CN, STATIONS_EN)):
        r = results[se]
        yte = r['yte']

        # Persistence
        ax = axes[0, i]
        dfs_s = r['dfs']
        valid_all = dfs_s['y_lag1'].notna()
        y_all = dfs_s['y'].values[valid_all]
        p_all = dfs_s['y_lag1'].values[valid_all]
        sp = r['split'] - dfs_s['y_lag1'].isna().sum()

        ax.scatter(y_all[:sp], p_all[:sp], c=C['train'], s=6, alpha=0.35, edgecolors='none', label='Train')
        ax.scatter(y_all[sp:], p_all[sp:], c=C['test'], s=14, alpha=0.6, edgecolors='none', label='Test')
        ax.plot([9,19],[9,19], 'k--', lw=0.7, alpha=0.3)
        r2_t = r2_score(y_all[sp:], p_all[sp:])
        ax.set_xlabel('ln(Observed)'); ax.set_ylabel('ln(Predicted)')
        ax.set_title(f'{chr(97+i)}  {se} \u2014 Persistence', fontweight='bold', loc='left', color=C['dark'])
        ax.text(0.95, 0.05, f'Test R\u00b2={r2_t:.2f}', transform=ax.transAxes,
               ha='right', va='bottom', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='#DDD'))
        if i == 0: ax.legend(fontsize=7, loc='upper left')
        ax.set_aspect('equal')

        # Best model
        ax = axes[1, i]
        if se == 'Lanshanzui': pred, mname = r['rf_pred'], 'RF Top5'
        else: pred, mname = r['ridge_pred'], 'Ridge Top5'

        ax.scatter(yte, pred, c=C['time'], s=14, alpha=0.65, edgecolors='#3D6B4F', lw=0.3)
        ax.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'k--', lw=0.7, alpha=0.3)
        r2_m = r2_score(yte, pred); rmse_m = np.sqrt(mean_squared_error(yte, pred))
        ax.set_xlabel('ln(Observed)'); ax.set_ylabel('ln(Predicted)')
        ax.set_title(f'{chr(100+i)}  {se} \u2014 {mname}', fontweight='bold', loc='left', color=C['dark'])
        ax.text(0.95, 0.05, f'R\u00b2={r2_m:.3f}  RMSE={rmse_m:.3f}',
               transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
               fontweight='bold', bbox=dict(boxstyle='round', fc='white', alpha=0.85, ec='#DDD'))
        ax.set_aspect('equal')

    fig.suptitle('Figure 2. Observed vs. predicted values on the test set',
                 fontweight='bold', fontsize=12, color=C['dark'], y=1.01)
    plt.tight_layout()
    fig.savefig('Figure2_FINAL.pdf')
    plt.close()
    print('Saved -> Figure2_FINAL.pdf')

# ============================================================
# FIGURE 3 — Feature importance + Ablation (FIXED overlap)
# ============================================================
def figure3():
    fig = plt.figure(figsize=(16, 10))

    # (a)(b)(c) Top 15 Gini
    for i, se in enumerate(STATIONS_EN):
        ax = fig.add_subplot(2, 3, i+1)
        r = results[se]
        gini = r['gini']
        idx_sorted = np.argsort(gini)[-15:][::-1]
        names = [feature_cols[j] for j in idx_sorted]
        vals  = [gini[j] for j in idx_sorted]
        clrs  = [C['y_lag'] if n.startswith('y_lag')
                 else C['time'] if ('month' in n or 'doy' in n)
                 else C['param'] for n in names]

        ax.barh(range(15), vals, color=clrs, height=0.65, edgecolor='white', lw=0.3)
        ax.set_yticks(range(15))
        ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel('Gini importance')
        ax.set_title(f'{chr(97+i)}  {se}', fontweight='bold', loc='left', color=C['dark'])
        if i == 0:
            leg = [Patch(color=C['y_lag'], label='y_lag (AR)'),
                   Patch(color=C['param'], label='Water quality'),
                   Patch(color=C['time'], label='Temporal')]
            ax.legend(handles=leg, fontsize=7, loc='lower right', framealpha=0.85)

    # (d) Category contribution
    ax = fig.add_subplot(2, 3, 4)
    x = np.arange(3); w = 0.22
    cat_keys = ['y_lag', 'param', 'time']
    cat_labels = ['Autoregressive', 'Water quality', 'Temporal']
    cat_colors = [C['y_lag'], C['param'], C['time']]
    for j, (label, clr, key) in enumerate(zip(cat_labels, cat_colors, cat_keys)):
        vals = [results[s]['cat_gini'][key]*100 for s in STATIONS_EN]
        bars = ax.bar(x + j*w, vals, w, color=clr, alpha=0.9, edgecolor='white', lw=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                   f'{v:.1f}%', ha='center', fontsize=7, fontweight='bold', color=C['dark'])
    ax.set_xticks(x + w); ax.set_xticklabels(STATIONS_EN, color=C['dark'])
    ax.set_ylabel('Gini importance share (%)')
    ax.set_title('(d) Category contribution (Gini)', fontweight='bold', loc='left', color=C['dark'])
    ax.set_ylim(0, 105)
    ax.legend([Patch(color=c) for c in cat_colors], cat_labels, fontsize=7, loc='upper right', framealpha=0.85)

    # (e) Ablation — FIXED: abbreviated names, no overlap
    ax = fig.add_subplot(2, 3, 5)
    ax.axis('off')
    ax.set_title('(e) Ablation: Top5 before vs. after adding y_lag4',
                fontweight='bold', fontsize=10, loc='center', color=C['dark'], y=1.02)

    before = {
        'Lanshanzui': ['y_lag1','y_lag2','y_lag3','y_lag5','y_lag7'],
        'Tuoshan':    ['y_lag1','y_lag2','y_lag3','y_lag5','doy_sin'],
        'Xidong':     ['y_lag1','y_lag2','y_lag3','doy_cos','CODMn_lag1'],
    }
    after = {
        'Lanshanzui': ['y_lag1','y_lag2','y_lag3','y_lag5','y_lag4'],
        'Tuoshan':    ['y_lag1','y_lag2','y_lag3','y_lag4','y_lag5'],
        'Xidong':     ['y_lag1','y_lag2','y_lag3','y_lag5','doy_cos'],
    }
    removed = {
        'Lanshanzui': ('y_lag7', C['y_lag']),
        'Tuoshan':    ('doy_sin', C['time']),
        'Xidong':     ('CODMn_lag1', C['param']),
    }

    # Draw each station as a compact feature-block row
    y_positions = [0.78, 0.50, 0.22]
    for i, se in enumerate(STATIONS_EN):
        y = y_positions[i]

        # Station name
        ax.text(0.02, y+0.06, se, fontsize=8, fontweight='bold', color=C['dark'], va='center')

        # "Before" label + feature pills
        ax.text(0.15, y+0.06, 'Before:', fontsize=7, color='#888', va='center')
        bx = 0.25
        for j, feat in enumerate(before[se]):
            cat = 'y_lag' if feat.startswith('y_lag') else 'time' if 'doy' in feat else 'param'
            color = C[cat]
            ax.text(bx + j*0.085, y+0.06, abbrev([feat])[0], fontsize=6.5, fontweight='bold',
                   color='white', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.25', fc=color, ec='none', alpha=0.85))

        # Arrow
        ax.annotate('', xy=(bx+5*0.085+0.02, y+0.06), xytext=(bx+5*0.085+0.06, y+0.06),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))

        # "After" label + feature pills
        ax.text(0.79, y+0.06, 'After:', fontsize=7, color='#888', va='center')
        ax_x = 0.86
        for j, feat in enumerate(after[se]):
            cat = 'y_lag' if feat.startswith('y_lag') else 'time' if 'doy' in feat else 'param'
            color = C[cat]
            ax.text(ax_x + j*0.085, y+0.06, abbrev([feat])[0], fontsize=6.5, fontweight='bold',
                   color='white', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.25', fc=color, ec='none', alpha=0.85))

        # Removed feature (in its own color)
        rem_feat, rem_color = removed[se]
        ax.text(0.02, y-0.06, f'Removed: ', fontsize=6.5, color='#888', va='center')
        ax.text(0.12, y-0.06, abbrev([rem_feat])[0], fontsize=7, fontweight='bold',
               color=rem_color, va='center',
               bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=rem_color, lw=1.2, alpha=0.9))

    # (f) Model comparison
    ax = fig.add_subplot(2, 3, 6)
    r2_data = {m: [results[s]['metrics'][m][0] for s in STATIONS_EN] for m in MODEL_ORDER}
    x = np.arange(3); w = 0.2
    for j, (model, clr) in enumerate(zip(MODEL_ORDER, MODEL_COLORS)):
        vals = r2_data[model]
        bars = ax.bar(x + j*w, vals, w, color=clr, alpha=0.88, edgecolor='white', lw=0.3)
        for bar, v in zip(bars, vals):
            va = 'bottom' if v >= 0 else 'top'; off = 0.025 if v >= 0 else -0.07
            ax.text(bar.get_x()+bar.get_width()/2, v+off, f'{v:.2f}',
                   ha='center', va=va, fontsize=6.5, fontweight='bold', color=C['dark'])
    ax.axhline(0, color='#AAA', lw=0.5)
    ax.set_xticks(x + 1.5*w); ax.set_xticklabels(STATIONS_EN, color=C['dark'])
    ax.set_ylabel('Test R\u00b2')
    ax.set_title('(f) Model performance', fontweight='bold', loc='left', color=C['dark'])
    ax.legend([Patch(color=c) for c in MODEL_COLORS],
             [m.replace('\n',' ') for m in MODEL_ORDER], fontsize=7, loc='lower right', framealpha=0.85)

    fig.suptitle('Figure 3. Feature importance (Gini), category contribution, and ablation study',
                 fontweight='bold', fontsize=12, color=C['dark'], y=1.01)
    plt.tight_layout()
    fig.savefig('Figure3_FINAL.pdf')
    plt.close()
    print('Saved -> Figure3_FINAL.pdf')

# ============================================================
# FIGURE 4 — No vertical text
# ============================================================
def figure4():
    fig = plt.figure(figsize=(16, 11))

    # (a) R2
    ax = fig.add_subplot(2, 3, 1)
    r2_data = {m: [results[s]['metrics'][m][0] for s in STATIONS_EN] for m in MODEL_ORDER}
    x = np.arange(3); w = 0.2
    for j, (model, clr) in enumerate(zip(MODEL_ORDER, MODEL_COLORS)):
        vals = r2_data[model]
        bars = ax.bar(x + j*w, vals, w, color=clr, alpha=0.88, edgecolor='white', lw=0.3)
        for bar, v in zip(bars, vals):
            va = 'bottom' if v>=0 else 'top'; off = 0.025 if v>=0 else -0.07
            ax.text(bar.get_x()+bar.get_width()/2, v+off, f'{v:.2f}',
                   ha='center', va=va, fontsize=6.5, fontweight='bold', color=C['dark'])
    ax.axhline(0, color='#AAA', lw=0.5)
    ax.set_xticks(x + 1.5*w); ax.set_xticklabels(STATIONS_EN, color=C['dark'])
    ax.set_ylabel('Test R\u00b2')
    ax.set_title('(a) Test R\u00b2 by model and station', fontweight='bold', loc='left', color=C['dark'])
    ax.legend([Patch(color=c) for c in MODEL_COLORS],
             [m.replace('\n',' ') for m in MODEL_ORDER], fontsize=7, loc='lower right', framealpha=0.85)

    # (b) RMSE — FIXED: no vertical text
    ax = fig.add_subplot(2, 3, 2)
    rmse_data = {m: [results[s]['metrics'][m][1] for s in STATIONS_EN] for m in MODEL_ORDER}
    max_rmse = max(max(v) for v in rmse_data.values())
    for j, (model, clr) in enumerate(zip(MODEL_ORDER, MODEL_COLORS)):
        vals = rmse_data[model]
        bars = ax.bar(x + j*w, vals, w, color=clr, alpha=0.88, edgecolor='white', lw=0.3)
        for bar, v in zip(bars, vals):
            # Always place text above the bar, never rotate
            offset = max_rmse * 0.03
            ax.text(bar.get_x()+bar.get_width()/2, v+offset, f'{v:.2f}',
                   ha='center', va='bottom', fontsize=6.5, fontweight='bold', color=C['dark'])
    ax.set_xticks(x + 1.5*w); ax.set_xticklabels(STATIONS_EN, color=C['dark'])
    ax.set_ylabel('RMSE (ln scale)')
    ax.set_title('(b) RMSE by model and station', fontweight='bold', loc='left', color=C['dark'])
    ax.set_ylim(0, max_rmse * 1.15)  # Extra room for labels

    # (c) Train-test gap
    ax = fig.add_subplot(2, 3, 3)
    tt_labels = []; train_vals = []; test_vals = []
    for se in STATIONS_EN:
        r = results[se]
        tt_labels.append(f'RF {se}'); train_vals.append(r['rf_train_r2'])
        test_vals.append(r['metrics']['RF Top5'][0])
        tt_labels.append(f'Ridge {se}'); train_vals.append(r['ridge_train_r2'])
        test_vals.append(r['metrics']['Ridge Top5'][0])

    ypos = range(len(tt_labels))
    ax.barh([y-0.15 for y in ypos], train_vals, 0.3, color=C['train'], alpha=0.75, label='Train R\u00b2')
    ax.barh([y+0.15 for y in ypos], test_vals,  0.3, color=C['test'], alpha=0.75, label='Test R\u00b2')
    for k in range(0, len(tt_labels), 2):
        gap = train_vals[k] - test_vals[k]
        mid = (train_vals[k] + test_vals[k]) / 2
        ax.annotate(f'gap={gap:.2f}', xy=(mid, ypos[k]), fontsize=7, ha='center', va='center',
                   color=C['rf'], fontweight='bold')
    ax.set_yticks(ypos); ax.set_yticklabels(tt_labels, fontsize=7, color=C['dark'])
    ax.axvline(0, color='#AAA', lw=0.5)
    ax.set_xlabel('R\u00b2')
    ax.set_title('(c) Train\u2013test R\u00b2 gap', fontweight='bold', loc='left', color=C['dark'])
    ax.legend(fontsize=7, loc='lower right', framealpha=0.85)

    # (d)(e)(f) Residuals
    for i, se in enumerate(STATIONS_EN):
        ax = fig.add_subplot(2, 3, 4+i)
        r = results[se]
        dfs_s = r['dfs']; split = r['split']

        if se == 'Lanshanzui': pred, mname = r['rf_pred'], 'RF'
        else: pred, mname = r['ridge_pred'], 'Ridge'

        res = r['yte'] - pred
        test_dates = dfs_s['Date'].values[split:]
        valid_te = ~np.isnan(dfs_s[feature_cols].values[split:]).any(axis=1)
        dates_clean = test_dates[valid_te]

        ax.scatter(dates_clean, res, c=C['dark'], s=10, alpha=0.5, edgecolors='none')
        ax.axhline(0, color=C['rf'], ls='--', lw=1.0, alpha=0.7)
        sd = np.std(res)
        ax.axhline(2*sd, color=C['persist'], ls=':', lw=0.5, alpha=0.5)
        ax.axhline(-2*sd, color=C['persist'], ls=':', lw=0.5, alpha=0.5)
        ax.set_xlabel('Date'); ax.set_ylabel('Residual (ln scale)')

        mu_val = np.mean(res)
        ax.set_title(f"({chr(100+i)}) {se} \u2014 {mname}   "
                    f"mean={mu_val:.3f}, std={sd:.3f}",
                    fontweight='bold', loc='left', fontsize=9, color=C['dark'])
        ax.set_ylim(-3*sd, 3*sd)

    fig.suptitle('Figure 4. Model comparison and residual diagnostics',
                 fontweight='bold', fontsize=12, color=C['dark'], y=1.01)
    plt.tight_layout()
    fig.savefig('Figure4_FINAL.pdf')
    plt.close()
    print('Saved -> Figure4_FINAL.pdf')

# ============================================================
# FIGURE 5 — Multi-metric panel
# ============================================================
def figure5():
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    configs = [
        ('R2', 0, 'Test R\u00b2', True, axes[0,0]),
        ('RMSE', 1, 'RMSE (ln scale)', False, axes[0,1]),
        ('MAE', 2, 'MAE (ln scale)', False, axes[1,0]),
        ('CV-R2', 3, 'CV-R\u00b2 (TimeSeries 5-fold)', True, axes[1,1]),
    ]

    x = np.arange(3); w = 0.2
    for metric, idx, ylabel, show_zero, ax in configs:
        for j, (model, clr) in enumerate(zip(MODEL_ORDER, MODEL_COLORS)):
            if metric == 'R2':
                vals = [results[s]['metrics'][model][0] for s in STATIONS_EN]
            elif metric == 'RMSE':
                vals = [results[s]['metrics'][model][1] for s in STATIONS_EN]
            elif metric == 'MAE':
                vals = [results[s]['metrics'][model][2] for s in STATIONS_EN]
            else:  # CV-R2
                vals = [np.mean(results[s]['metrics'][model][3]) for s in STATIONS_EN]
                errs = [np.std(results[s]['metrics'][model][3]) for s in STATIONS_EN]

            bars = ax.bar(x + j*w, vals, w, color=clr, alpha=0.88, edgecolor='white', lw=0.3)
            for k, (bar, v) in enumerate(zip(bars, vals)):
                if metric == 'R2':
                    va = 'bottom' if v >= 0 else 'top'
                    off = 0.025 if v >= 0 else -0.07
                else:
                    va = 'bottom'; off = 0.015
                ax.text(bar.get_x()+bar.get_width()/2, v+off, f'{v:.2f}',
                       ha='center', va=va, fontsize=6.5, fontweight='bold', color=C['dark'])

            if metric == 'CV-R2':
                ax.errorbar(x + j*w, vals, yerr=errs, fmt='none',
                           ecolor=C['dark'], capsize=3, lw=1.0, alpha=0.5)

        if show_zero: ax.axhline(0, color='#AAA', lw=0.5)
        ax.set_xticks(x + 1.5*w); ax.set_xticklabels(STATIONS_EN, color=C['dark'])
        ax.set_ylabel(ylabel)
        ax.set_title(f'({chr(97+["R2","RMSE","MAE","CV-R2"].index(metric))}) {ylabel}',
                    fontweight='bold', loc='left', color=C['dark'])

    fig.legend([Patch(color=c) for c in MODEL_COLORS],
              [m.replace('\n',' ') for m in MODEL_ORDER],
              loc='upper center', ncol=4, fontsize=8, framealpha=0.85,
              bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('Figure 5. Multi-metric model comparison across three stations',
                 fontweight='bold', fontsize=12, color=C['dark'], y=1.06)
    plt.tight_layout()
    fig.savefig('Figure5_FINAL.pdf')
    plt.close()
    print('Saved -> Figure5_FINAL.pdf')

# ============================================================
if __name__ == '__main__':
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print('\nAll 5 figures saved.')