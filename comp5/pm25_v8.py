"""
V8: Key insight — PM10 (r=0.93) is available at test time, no recursion.
PM2.5 lag-1 (r=0.96 in train) degrades to ~0.80 due to recursive errors.
Fix: Remove short PM2.5 lags (1-12h), rely on PM10/CO as exact proxies.
Keep only PM2.5 lags 24h+ (less recursive contamination).
Add historical PM2.5 by (month, hour) from training as stable seasonality feature.
Model: HGB only (was best). 5-fold, fold-5 RMSE as primary metric.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os, time, json, shutil
from collections import deque
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT    = r"C:\Users\jaish\OneDrive - The University of Texas at Austin\Documents\AfterQueryMLComps\comp5"
DL     = r"C:\Users\jaish\Downloads"
LOG    = os.path.join(OUT, "v8_log.txt")
SUB    = os.path.join(DL,  "submission_v8.csv")

open(LOG,'w').close()
T0 = time.time()
def log(m):
    msg = f"[{(time.time()-T0)/60:5.1f}m] {m}"
    print(msg, flush=True); open(LOG,'a').write(msg+'\n')

def rmse(y_orig, y_log):
    return float(np.sqrt(mean_squared_error(y_orig, np.expm1(y_log).clip(0))))

# ── Data ──────────────────────────────────────────────────────────────────────
log("Loading data...")
DL_TRAIN = os.path.join(DL, "train (5).csv")
DL_TEST  = os.path.join(DL, "test (5).csv")
train = pd.read_csv(DL_TRAIN, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
test  = pd.read_csv(DL_TEST,  parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
train['pm25_orig'] = train['pm25'].clip(lower=0)
train['pm25_log']  = np.log1p(train['pm25_orig'])

# ── Historical PM2.5 stats (from training only — no leakage) ──────────────────
log("Computing historical PM2.5 stats from training...")
hist = {}
hist['month_hour']    = train.groupby(['month','hour'])['pm25_orig'].mean().rename('hist_month_hour')
hist['month']         = train.groupby('month')['pm25_orig'].mean().rename('hist_month')
hist['hour']          = train.groupby('hour')['pm25_orig'].mean().rename('hist_hour')
hist['month_weekend'] = train.groupby(['month', train['datetime'].dt.dayofweek.ge(5).astype(int)])['pm25_orig'].mean().rename('hist_month_weekend')

def add_hist_features(df):
    df = df.copy()
    dow = df['datetime'].dt.dayofweek
    df['is_weekend'] = (dow >= 5).astype(int)
    df = df.merge(hist['month_hour'].reset_index(), on=['month','hour'], how='left')
    df = df.merge(hist['month'].reset_index(),      on='month',          how='left')
    df = df.merge(hist['hour'].reset_index(),       on='hour',           how='left')
    df['hist_month_weekend'] = df.apply(
        lambda r: hist['month_weekend'].get((r['month'], r['is_weekend']), np.nan), axis=1)
    # Deviation ratios (how much does this hour/month deviate from annual mean)
    annual_mean = train['pm25_orig'].mean()
    df['hist_month_ratio']     = df['hist_month']      / annual_mean
    df['hist_hour_ratio']      = df['hist_hour']       / annual_mean
    df['hist_month_hour_ratio']= df['hist_month_hour'] / annual_mean
    return df

train = add_hist_features(train)
test  = add_hist_features(test)

# ── Temporal features ─────────────────────────────────────────────────────────
WIND = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,
        'SSE':157.5,'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,
        'WNW':292.5,'NW':315,'NNW':337.5}

def add_temporal(df):
    df = df.copy()
    df['wind_dir_deg'] = df['wind_direction'].map(WIND)
    df['wind_dir_rad'] = np.radians(df['wind_dir_deg'])
    df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
    df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
    df['hour_sin']  = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']  = np.cos(2*np.pi*df['hour']/24)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    dow = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2*np.pi*dow/7)
    df['dow_cos'] = np.cos(2*np.pi*dow/7)
    doy = df['datetime'].dt.dayofyear
    df['doy_sin'] = np.sin(2*np.pi*doy/365)
    df['doy_cos'] = np.cos(2*np.pi*doy/365)
    df['is_winter']         = df['month'].isin([11,12,1,2,3]).astype(np.int8)
    df['is_heating_season'] = df['month'].isin([11,12,1,2]).astype(np.int8)
    df['year_frac']         = df['year'] + (doy-1)/365.0
    return df

train = add_temporal(train); test = add_temporal(test)

# ── Combined exog interpolation ───────────────────────────────────────────────
BASE = ['pm10','so2','no2','co','o3','temperature','pressure','dew_point','wind_speed','rain']
tc   = train.drop(columns=['pm25','pm25_orig','pm25_log']); tc['_t'] = False
tec  = test.copy(); tec['_t'] = True
comb = pd.concat([tc, tec], ignore_index=True).sort_values('datetime').reset_index(drop=True)

for col in BASE + ['wind_dir_deg','wind_dir_rad','wind_dir_sin','wind_dir_cos']:
    comb[col] = comb[col].interpolate(method='linear', limit=6, limit_direction='both').ffill().bfill()

# Interactions
comb['humidity_proxy'] = comb['dew_point'] - comb['temperature']
comb['wind_u']         = comb['wind_speed'] * np.sin(comb['wind_dir_rad'])
comb['wind_v']         = comb['wind_speed'] * np.cos(comb['wind_dir_rad'])
comb['atm_stability']  = comb['temperature'] - comb['dew_point']
comb['calm_flag']      = (comb['wind_speed'] < 1.0).astype(np.int8)
comb['pm10_x_no2']     = comb['pm10'] * comb['no2']
comb['no2_x_co']       = comb['no2'] * comb['co']
comb['pollution_load'] = comb['pm10'] + comb['no2'] + comb['so2']
comb['pm10_per_wind']  = comb['pm10'] / (comb['wind_speed'].clip(lower=0.1))
comb['co_per_wind']    = comb['co']  / (comb['wind_speed'].clip(lower=0.1))
comb['temp_x_winter']  = comb['temperature'] * comb['is_winter']
for h in [1,3,6,12,24]:
    comb[f'pressure_d{h}'] = comb['pressure'] - comb['pressure'].shift(h)
    comb[f'temp_d{h}']     = comb['temperature'] - comb['temperature'].shift(h)
    comb[f'pm10_d{h}']     = comb['pm10'] - comb['pm10'].shift(h)
    comb[f'co_d{h}']       = comb['co'] - comb['co'].shift(h)

# ── Exogenous lags (all from test.csv, no recursion needed) ───────────────────
# KEY STRATEGY: Use PM10 and CO very heavily — they're actual measurements at test time
# PM10 r=0.93, CO r=0.83 with PM2.5
LAG_H  = [1,2,3,6,12,24,48,72,168,336]
ROLL_W = [3,6,12,24,48,168]
EW_H   = [3,6,12,24,72]

# HIGH-PRIORITY: pm10 and co get dense lags (best proxies for PM2.5)
for col in ['pm10', 'co', 'no2']:
    for lag in LAG_H:
        comb[f'{col}_lag{lag}'] = comb[col].shift(lag)
    for w in ROLL_W:
        r = comb[col].rolling(w, min_periods=1)
        comb[f'{col}_r{w}_mean'] = r.mean()
        comb[f'{col}_r{w}_std']  = r.std()
        comb[f'{col}_r{w}_min']  = r.min()
        comb[f'{col}_r{w}_max']  = r.max()
    for hl in EW_H:
        comb[f'{col}_ew{hl}'] = comb[col].ewm(halflife=hl, min_periods=1).mean()

# MEDIUM: other exog cols
for col in ['so2','o3','temperature','pressure','dew_point','wind_speed','rain',
            'pm10_per_wind','co_per_wind','pollution_load','humidity_proxy']:
    for lag in [1,3,6,12,24,48,168]:
        comb[f'{col}_lag{lag}'] = comb[col].shift(lag)
    for w in [6,24,168]:
        r = comb[col].rolling(w, min_periods=1)
        comb[f'{col}_r{w}_mean'] = r.mean()
        comb[f'{col}_r{w}_std']  = r.std()
    for hl in [6,24,72]:
        comb[f'{col}_ew{hl}'] = comb[col].ewm(halflife=hl, min_periods=1).mean()

log(f"Exog features built. comb shape: {comb.shape}")

# ── Split ─────────────────────────────────────────────────────────────────────
trn_e = comb[~comb['_t']].drop(columns=['_t']).reset_index(drop=True)
tst_e = comb[ comb['_t']].drop(columns=['_t']).reset_index(drop=True)
trn_e['pm25_orig'] = train['pm25_orig'].values
trn_e['pm25_log']  = train['pm25_log'].values

# ── PM2.5 AR features: ONLY long lags (24h+) to reduce recursive contamination ─
pm25_s = train['pm25_orig'].copy()
# Long lags only — these are less contaminated:
# At test step i, lag-24 was predicted 24 steps ago (reasonable accuracy)
# lag-168 was predicted 168 steps ago (many steps back but model had time to stabilize)
PM25_LAGS = [24, 48, 72, 168, 336]          # NO short lags
PM25_ROLL = [24, 48, 168]                    # NO short rolling windows
PM25_EW   = [24, 72]                         # NO short EWMA

for lag in PM25_LAGS:
    trn_e[f'pm25_lag{lag}'] = pm25_s.shift(lag).values
for w in PM25_ROLL:
    r = pm25_s.shift(1).rolling(w, min_periods=1)
    trn_e[f'pm25_r{w}_mean'] = r.mean().values
    trn_e[f'pm25_r{w}_std']  = r.std().values
for hl in PM25_EW:
    trn_e[f'pm25_ew{hl}'] = pm25_s.shift(1).ewm(halflife=hl, min_periods=1).mean().values

# Placeholders for test (filled recursively)
for lag in PM25_LAGS:
    tst_e[f'pm25_lag{lag}'] = np.nan
for w in PM25_ROLL:
    tst_e[f'pm25_r{w}_mean'] = np.nan
    tst_e[f'pm25_r{w}_std']  = np.nan
for hl in PM25_EW:
    tst_e[f'pm25_ew{hl}'] = np.nan

EXCL  = {'record_id','datetime','pm25','pm25_orig','pm25_log','_t',
         'wind_direction','wind_dir_deg','wind_dir_rad'}
FCOLS = [c for c in trn_e.columns if c not in EXCL]
log(f"Total features: {len(FCOLS)}  (PM2.5 lags: {len(PM25_LAGS)+2*len(PM25_ROLL)+len(PM25_EW)})")

X      = trn_e[FCOLS].values.astype(np.float32)
y      = trn_e['pm25_log'].values
yo     = trn_e['pm25_orig'].values
X_test = tst_e[FCOLS].values.astype(np.float32)

feat_idx     = {col: i for i, col in enumerate(FCOLS)}
PM25_LAG_IDX = [(lag, feat_idx[f'pm25_lag{lag}']) for lag in PM25_LAGS]
PM25_ROL_IDX = [(w, s, feat_idx[f'pm25_r{w}_{s}']) for w in PM25_ROLL for s in ['mean','std']]
PM25_EW_IDX  = [(hl, feat_idx[f'pm25_ew{hl}']) for hl in PM25_EW]

# ── HGB params (best in V7) ────────────────────────────────────────────────────
HGB_P = dict(max_iter=600, learning_rate=0.02, max_leaf_nodes=127,
             min_samples_leaf=20, l2_regularization=0.3,
             max_depth=8, early_stopping=False, random_state=42)

# Also train a LGB for blending (V2 params)
try:
    v2s = optuna.load_study(study_name='lgbm_v2', storage=f'sqlite:///{OUT}/study_v2.db')
    bp  = v2s.best_params
except: bp = {}
LGB_P = dict(
    learning_rate=bp.get('lr',0.04), num_leaves=int(bp.get('nl',349)),
    min_child_samples=int(bp.get('mcs',32)), feature_fraction=bp.get('ff',0.95),
    bagging_fraction=bp.get('bf',0.84), reg_alpha=bp.get('ra',0.01),
    reg_lambda=bp.get('rl',0.16), min_split_gain=bp.get('msg',0.04),
    n_estimators=2500, bagging_freq=1, n_jobs=-1, random_state=42, verbose=-1
)

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
oof_hgb = np.full(len(y), np.nan)
oof_lgb = np.full(len(y), np.nan)
lgb_iters = []
log("\n=== 5-fold CV ===")

for fold, (tri, vali) in enumerate(tscv.split(X)):
    ft = time.time()
    mh = HistGradientBoostingRegressor(**HGB_P)
    mh.fit(X[tri], y[tri]); oof_hgb[vali] = mh.predict(X[vali])
    rh = rmse(yo[vali], oof_hgb[vali])

    ml = lgb.LGBMRegressor(**LGB_P)
    ml.fit(X[tri], y[tri], eval_set=[(X[vali], y[vali])],
           callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(-1)])
    oof_lgb[vali] = ml.predict(X[vali]); lgb_iters.append(ml.best_iteration_)
    rl = rmse(yo[vali], oof_lgb[vali])

    log(f"  fold {fold+1}: HGB={rh:.4f}  LGB={rl:.4f}(n={ml.best_iteration_})  ({time.time()-ft:.0f}s)")

mask = ~(np.isnan(oof_hgb)|np.isnan(oof_lgb))
r_hgb = rmse(yo[mask], oof_hgb[mask])
r_lgb = rmse(yo[mask], oof_lgb[mask])
log(f"\n  HGB OOF: {r_hgb:.4f}  LGB OOF: {r_lgb:.4f}")

# Blend
def obj(t):
    wh = t.suggest_float('wh',0,1)
    return rmse(yo[mask], wh*oof_hgb[mask]+(1-wh)*oof_lgb[mask])
bs = optuna.create_study(direction='minimize'); bs.optimize(obj,n_trials=200,n_jobs=1)
wh = bs.best_params['wh']; wl = 1-wh
log(f"  Blend OOF: {bs.best_value:.4f}  wHGB={wh:.3f} wLGB={wl:.3f}")

# ── Final training ────────────────────────────────────────────────────────────
log("\nFinal training on all data...")
hgb_f = HistGradientBoostingRegressor(**HGB_P); hgb_f.fit(X, y); log("  HGB done")

n_lgb = max(150, int(np.mean([b for b in lgb_iters if b and b>0])))
LGB_P['n_estimators'] = n_lgb
lgb_f = lgb.LGBMRegressor(**LGB_P); lgb_f.fit(X, y); log(f"  LGB done (n={n_lgb})")

# ── Recursive test prediction (only long PM2.5 lags to recurse) ──────────────
log("\nRecursive test prediction (long-lag AR only)...")
ewma_alphas = {hl: 1-np.exp(-np.log(2)/hl) for hl in PM25_EW}
ewma_state  = {}
for hl in PM25_EW:
    a = ewma_alphas[hl]; v = float(train['pm25_orig'].iloc[0])
    for val in train['pm25_orig'].values[1:]: v = a*float(val) + (1-a)*v
    ewma_state[hl] = v

pm25_buf = deque(train['pm25_orig'].values, maxlen=400)
preds_log = []; t0 = time.time()

for i in range(len(X_test)):
    row = X_test[i].copy()
    buf = list(pm25_buf)
    for lag, idx in PM25_LAG_IDX:
        row[idx] = buf[-lag] if len(buf) >= lag else np.nan
    for w_r, stat, idx in PM25_ROL_IDX:
        rec = buf[-w_r:] if len(buf) >= w_r else buf
        if not rec: row[idx] = np.nan; continue
        row[idx] = float(np.mean(rec)) if stat=='mean' else (float(np.std(rec)) if len(rec)>1 else 0.)
    for hl, idx in PM25_EW_IDX:
        row[idx] = ewma_state[hl]

    r2 = row.reshape(1,-1)
    p  = wh*hgb_f.predict(r2)[0] + wl*lgb_f.predict(r2)[0]
    preds_log.append(p)
    pp = float(np.expm1(p).clip(0))
    pm25_buf.append(pp)
    for hl in PM25_EW: ewma_state[hl] = ewma_alphas[hl]*pp + (1-ewma_alphas[hl])*ewma_state[hl]
    if (i+1)%1000==0: log(f"  {i+1}/{len(X_test)} ({time.time()-t0:.0f}s) last={pp:.1f}")

log(f"  Recursive done in {time.time()-t0:.0f}s")

preds = np.expm1(np.array(preds_log)).clip(0)
sub = test[['record_id']].copy()
sub['predicted_pm25'] = preds
sub.to_csv(SUB, index=False)
shutil.copy(SUB, os.path.join(OUT, "submission_v8.csv"))

log(f"\n{'='*60}")
log(f"V8 COMPLETE  ({(time.time()-T0)/60:.1f} min total)")
log(f"  HGB OOF: {r_hgb:.4f}  LGB OOF: {r_lgb:.4f}  Blend: {bs.best_value:.4f}")
log(f"  Features: {len(FCOLS)}  PM2.5 lags: {PM25_LAGS}")
log(f"  Pred: min={preds.min():.1f}  mean={preds.mean():.1f}  max={preds.max():.1f}")
log(f"  Saved: {SUB}")
log(f"{'='*60}")
json.dump(dict(hgb=r_hgb, lgb=r_lgb, blend=bs.best_value, wh=wh, wl=wl,
               n_features=len(FCOLS), pm25_lags=PM25_LAGS),
          open(os.path.join(OUT,'v8_meta.json'),'w'), indent=2)
