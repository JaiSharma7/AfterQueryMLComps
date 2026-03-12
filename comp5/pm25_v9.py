"""
V9: Key fix — train on sqrt(PM2.5) target instead of log1p.
log1p causes systematic underestimation (Jensen's inequality) when scoring raw RMSE.
sqrt is less distorting: range sqrt(3)=1.7 to sqrt(898)=30, much better balanced.
Keep V7's feature set (all PM2.5 lags including short ones).
Add bias correction post-processing.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os, time, json, shutil
from collections import deque
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT  = r"C:\Users\jaish\OneDrive - The University of Texas at Austin\Documents\AfterQueryMLComps\comp5"
DL   = r"C:\Users\jaish\Downloads"
LOG  = os.path.join(OUT, "v9_log.txt")
SUB  = os.path.join(DL,  "submission_v9.csv")

open(LOG,'w').close()
T0 = time.time()
def log(m):
    msg = f"[{(time.time()-T0)/60:5.1f}m] {m}"
    print(msg, flush=True); open(LOG,'a').write(msg+'\n')

# ── Target helpers ─────────────────────────────────────────────────────────────
# Train on sqrt, score raw RMSE — much closer to actual eval metric than log1p
def to_target(pm25_raw): return np.sqrt(pm25_raw.clip(0))
def from_target(t_pred): return np.square(t_pred).clip(0)
def rmse_raw(y_orig, y_pred_raw):
    return float(np.sqrt(mean_squared_error(y_orig, y_pred_raw)))

# ── Data ──────────────────────────────────────────────────────────────────────
log("Loading data...")
train = pd.read_csv(os.path.join(DL,"train (5).csv"), parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
test  = pd.read_csv(os.path.join(DL,"test (5).csv"),  parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)
train['pm25_orig']  = train['pm25'].clip(lower=0)
train['pm25_sqrt']  = to_target(train['pm25_orig'])

# ── Historical PM2.5 stats (training only) ────────────────────────────────────
hist_mh = train.groupby(['month','hour'])['pm25_orig'].mean()
hist_m  = train.groupby('month')['pm25_orig'].mean()
pm25_pm10_ratio_mh = (train.groupby(['month','hour'])['pm25_orig'].mean() /
                      train.groupby(['month','hour'])['pm10'].mean().clip(lower=1))

def add_hist(df, is_test=False):
    df = df.copy()
    df['hist_pm25_month_hour'] = df.apply(lambda r: hist_mh.get((r['month'],r['hour']), hist_m.get(r['month'], train['pm25_orig'].mean())), axis=1)
    df['hist_pm25_month']      = df['month'].map(hist_m).fillna(train['pm25_orig'].mean())
    df['pm25_pm10_ratio_mh']   = df.apply(lambda r: pm25_pm10_ratio_mh.get((r['month'],r['hour']), 0.85), axis=1)
    return df

train = add_hist(train); test = add_hist(test)

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
tc   = train.drop(columns=['pm25','pm25_orig','pm25_sqrt']); tc['_t'] = False
tec  = test.copy(); tec['_t'] = True
comb = pd.concat([tc, tec], ignore_index=True).sort_values('datetime').reset_index(drop=True)

for col in BASE + ['wind_dir_deg','wind_dir_rad','wind_dir_sin','wind_dir_cos']:
    comb[col] = comb[col].interpolate(method='linear', limit=6, limit_direction='both').ffill().bfill()

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

# Exogenous lags (ALL available at test time from test.csv)
LAG_H  = [1,2,3,6,12,24,48,72,168,336]
ROLL_W = [3,6,12,24,48,168]
EW_H   = [3,6,12,24,72]
EXOG   = BASE + ['pm10_per_wind','co_per_wind','pollution_load','humidity_proxy','atm_stability']

for col in EXOG:
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

log(f"Features built. comb: {comb.shape}")

trn_e = comb[~comb['_t']].drop(columns=['_t']).reset_index(drop=True)
tst_e = comb[ comb['_t']].drop(columns=['_t']).reset_index(drop=True)
trn_e['pm25_orig'] = train['pm25_orig'].values
trn_e['pm25_sqrt'] = train['pm25_sqrt'].values

# PM2.5 AR features — ALL LAGS (same as V7, best model for recursion)
pm25_s = train['pm25_orig'].copy()
PM25_LAGS = [1,2,3,6,12,24,48,72,168,336]
PM25_ROLL = [3,6,12,24,48,168]
PM25_EW   = [3,6,12,24,72]

for lag in PM25_LAGS:
    trn_e[f'pm25_lag{lag}'] = pm25_s.shift(lag).values
for w in PM25_ROLL:
    r = pm25_s.shift(1).rolling(w, min_periods=1)
    trn_e[f'pm25_r{w}_mean'] = r.mean().values
    trn_e[f'pm25_r{w}_std']  = r.std().values
    trn_e[f'pm25_r{w}_min']  = r.min().values
    trn_e[f'pm25_r{w}_max']  = r.max().values
for hl in PM25_EW:
    trn_e[f'pm25_ew{hl}'] = pm25_s.shift(1).ewm(halflife=hl, min_periods=1).mean().values

for lag in PM25_LAGS:
    tst_e[f'pm25_lag{lag}'] = np.nan
for w in PM25_ROLL:
    for s in ['mean','std','min','max']:
        tst_e[f'pm25_r{w}_{s}'] = np.nan
for hl in PM25_EW:
    tst_e[f'pm25_ew{hl}'] = np.nan

EXCL  = {'record_id','datetime','pm25','pm25_orig','pm25_sqrt','_t',
         'wind_direction','wind_dir_deg','wind_dir_rad'}
FCOLS = [c for c in trn_e.columns if c not in EXCL]
log(f"Features: {len(FCOLS)}")

X      = trn_e[FCOLS].values.astype(np.float32)
y      = trn_e['pm25_sqrt'].values           # ← SQRT target
yo     = trn_e['pm25_orig'].values
X_test = tst_e[FCOLS].values.astype(np.float32)

feat_idx     = {col: i for i, col in enumerate(FCOLS)}
PM25_LAG_IDX = [(lag, feat_idx[f'pm25_lag{lag}']) for lag in PM25_LAGS]
PM25_ROL_IDX = [(w, s, feat_idx[f'pm25_r{w}_{s}']) for w in PM25_ROLL for s in ['mean','std','min','max']]
PM25_EW_IDX  = [(hl, feat_idx[f'pm25_ew{hl}']) for hl in PM25_EW]

# ── Model params ──────────────────────────────────────────────────────────────
try:
    v2s = optuna.load_study(study_name='lgbm_v2', storage=f'sqlite:///{OUT}/study_v2.db')
    bp  = v2s.best_params
except: bp = {}

# HGB tuned for raw-RMSE optimization
HGB_P = dict(max_iter=800, learning_rate=0.015, max_leaf_nodes=127,
             min_samples_leaf=20, l2_regularization=0.2,
             max_depth=8, early_stopping=False, random_state=42)

LGB_P = dict(
    learning_rate=bp.get('lr',0.04), num_leaves=int(bp.get('nl',349)),
    min_child_samples=int(bp.get('mcs',32)), feature_fraction=bp.get('ff',0.95),
    bagging_fraction=bp.get('bf',0.84), reg_alpha=bp.get('ra',0.01),
    reg_lambda=bp.get('rl',0.16), min_split_gain=bp.get('msg',0.04),
    n_estimators=2500, bagging_freq=1, n_jobs=-1, random_state=42, verbose=-1
)

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
oof_h = np.full(len(y), np.nan)
oof_l = np.full(len(y), np.nan)
lgb_iters = []
log("\n=== 5-fold CV (sqrt target) ===")

for fold, (tri, vali) in enumerate(tscv.split(X)):
    ft = time.time()
    mh = HistGradientBoostingRegressor(**HGB_P)
    mh.fit(X[tri], y[tri]); oof_h[vali] = mh.predict(X[vali])
    rh = rmse_raw(yo[vali], from_target(oof_h[vali]))

    ml = lgb.LGBMRegressor(**LGB_P)
    ml.fit(X[tri], y[tri], eval_set=[(X[vali], y[vali])],
           callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(-1)])
    oof_l[vali] = ml.predict(X[vali]); lgb_iters.append(ml.best_iteration_)
    rl = rmse_raw(yo[vali], from_target(oof_l[vali]))

    log(f"  fold {fold+1}: HGB={rh:.4f}  LGB={rl:.4f}(n={ml.best_iteration_})  ({time.time()-ft:.0f}s)")

mask = ~(np.isnan(oof_h)|np.isnan(oof_l))
r_hgb = rmse_raw(yo[mask], from_target(oof_h[mask]))
r_lgb = rmse_raw(yo[mask], from_target(oof_l[mask]))
log(f"\n  HGB OOF: {r_hgb:.4f}  LGB OOF: {r_lgb:.4f}")

def obj(t):
    wh = t.suggest_float('wh',0,1)
    return rmse_raw(yo[mask], from_target(wh*oof_h[mask]+(1-wh)*oof_l[mask]))
bs = optuna.create_study(direction='minimize'); bs.optimize(obj,n_trials=200,n_jobs=1)
wh = bs.best_params['wh']; wl = 1-wh
log(f"  Blend OOF: {bs.best_value:.4f}  wHGB={wh:.3f} wLGB={wl:.3f}")

# Bias correction factor from OOF
oof_blend_raw = from_target(wh*oof_h[mask] + wl*oof_l[mask])
bias_correction = yo[mask].mean() / oof_blend_raw.mean()
log(f"  Bias correction: {bias_correction:.4f}  (mean_actual={yo[mask].mean():.1f}  mean_pred={oof_blend_raw.mean():.1f})")

# Apply bias correction and re-score
oof_corrected = oof_blend_raw * bias_correction
r_corrected = rmse_raw(yo[mask], oof_corrected)
log(f"  After bias correction OOF: {r_corrected:.4f}")

# ── Final training ────────────────────────────────────────────────────────────
log("\nFinal training on all data...")
hgb_f = HistGradientBoostingRegressor(**HGB_P); hgb_f.fit(X, y); log("  HGB done")
n_lgb = max(150, int(np.mean([b for b in lgb_iters if b and b>0])))
LGB_P['n_estimators'] = n_lgb
lgb_f = lgb.LGBMRegressor(**LGB_P); lgb_f.fit(X, y); log(f"  LGB done (n={n_lgb})")

# ── Recursive test prediction with PM10 physical constraint ──────────────────
log("\nRecursive test prediction...")
ewma_alphas = {hl: 1-np.exp(-np.log(2)/hl) for hl in PM25_EW}
ewma_state  = {}
for hl in PM25_EW:
    a = ewma_alphas[hl]; v = float(train['pm25_orig'].iloc[0])
    for val in train['pm25_orig'].values[1:]: v = a*float(val)+(1-a)*v
    ewma_state[hl] = v

pm25_buf = deque(train['pm25_orig'].values, maxlen=400)
preds_raw = []; t0 = time.time()

pm10_vals = tst_e['pm10'].values  # actual PM10 measurements — always available

for i in range(len(X_test)):
    row = X_test[i].copy()
    buf = list(pm25_buf)
    for lag, idx in PM25_LAG_IDX:
        row[idx] = buf[-lag] if len(buf) >= lag else np.nan
    for w_r, stat, idx in PM25_ROL_IDX:
        rec = buf[-w_r:] if len(buf) >= w_r else buf
        if not rec: row[idx] = np.nan; continue
        if   stat=='mean': row[idx] = float(np.mean(rec))
        elif stat=='std':  row[idx] = float(np.std(rec)) if len(rec)>1 else 0.
        elif stat=='min':  row[idx] = float(np.min(rec))
        elif stat=='max':  row[idx] = float(np.max(rec))
    for hl, idx in PM25_EW_IDX:
        row[idx] = ewma_state[hl]

    r2 = row.reshape(1,-1)
    p_sqrt = wh*hgb_f.predict(r2)[0] + wl*lgb_f.predict(r2)[0]
    pp = float(from_target(np.array([p_sqrt]))[0])

    # Apply bias correction
    pp = pp * bias_correction

    # Physical constraint: PM2.5 ≤ PM10 (always true by definition)
    pm10_now = float(pm10_vals[i])
    pp = float(np.clip(pp, 0, pm10_now * 1.05))  # small buffer for measurement noise

    preds_raw.append(pp)
    pm25_buf.append(pp)
    for hl in PM25_EW: ewma_state[hl] = ewma_alphas[hl]*pp + (1-ewma_alphas[hl])*ewma_state[hl]
    if (i+1)%1000==0: log(f"  {i+1}/{len(X_test)} ({time.time()-t0:.0f}s) last={pp:.1f}")

log(f"  Done in {time.time()-t0:.0f}s")

preds = np.array(preds_raw).clip(0)
sub = test[['record_id']].copy(); sub['predicted_pm25'] = preds
sub.to_csv(SUB, index=False)
shutil.copy(SUB, os.path.join(OUT,"submission_v9.csv"))

log(f"\n{'='*60}")
log(f"V9 COMPLETE  ({(time.time()-T0)/60:.1f} min total)")
log(f"  HGB OOF: {r_hgb:.4f}  LGB OOF: {r_lgb:.4f}  Blend: {bs.best_value:.4f}")
log(f"  Bias correction: {bias_correction:.4f}  After correction: {r_corrected:.4f}")
log(f"  Pred: min={preds.min():.1f}  mean={preds.mean():.1f}  max={preds.max():.1f}")
log(f"  Saved: {SUB}")
log(f"{'='*60}")
json.dump(dict(hgb=r_hgb, lgb=r_lgb, blend=bs.best_value, corrected=r_corrected,
               wh=wh, wl=wl, bias_correction=bias_correction,
               n_features=len(FCOLS)), open(os.path.join(OUT,'v9_meta.json'),'w'), indent=2)
