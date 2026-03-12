"""
V7-Fast: LGB + HGB, 3-fold TSS, trains AR and No-AR simultaneously.
No XGB. No Optuna. Fixed equal blend. ~20min total.
Target: OOF < 16.5 (both AR and No-AR for LB reliability).
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
LOG    = os.path.join(OUT, "v7_log.txt")
SUB_V7 = os.path.join(DL,  "submission_v7.csv")
SUB_AR = os.path.join(DL,  "submission_v7_ar.csv")

open(LOG,'w').close()
t_start = time.time()
def log(m):
    elapsed = (time.time()-t_start)/60
    msg = f"[{elapsed:5.1f}m] {m}"
    print(msg, flush=True)
    open(LOG,'a').write(msg+'\n')

def rmse_orig(y_true_orig, y_pred_log):
    return float(np.sqrt(mean_squared_error(y_true_orig, np.expm1(y_pred_log).clip(0))))

# ── Load data ─────────────────────────────────────────────────────────────────
log("Loading data...")
train = pd.read_csv(os.path.join(DL,"train (5).csv"), parse_dates=['datetime'])
test  = pd.read_csv(os.path.join(DL,"test (5).csv"),  parse_dates=['datetime'])
train = train.sort_values('datetime').reset_index(drop=True)
test  = test.sort_values('datetime').reset_index(drop=True)
train['pm25_orig'] = train['pm25'].clip(lower=0)
train['pm25_log']  = np.log1p(train['pm25_orig'])
log(f"Train={len(train)}  Test={len(test)}")

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
    df['is_weekend']        = (dow >= 5).astype(np.int8)
    df['is_rush_hour']      = df['hour'].isin([7,8,9,17,18,19]).astype(np.int8)
    df['is_winter']         = df['month'].isin([11,12,1,2,3]).astype(np.int8)
    df['is_heating_season'] = df['month'].isin([11,12,1,2]).astype(np.int8)
    df['quarter']           = df['datetime'].dt.quarter
    df['year_frac']         = df['year'] + (doy-1)/365.0
    return df

train = add_temporal(train); test = add_temporal(test)

# ── Combined interpolation ────────────────────────────────────────────────────
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
comb['very_calm']      = (comb['wind_speed'] < 0.3).astype(np.int8)
comb['pm10_x_no2']     = comb['pm10'] * comb['no2']
comb['no2_x_co']       = comb['no2'] * comb['co']
comb['so2_x_co']       = comb['so2'] * comb['co']
comb['pollution_load'] = comb['pm10'] + comb['no2'] + comb['so2']
comb['pm10_per_wind']  = comb['pm10'] / (comb['wind_speed'].clip(lower=0.1))
comb['co_per_wind']    = comb['co']  / (comb['wind_speed'].clip(lower=0.1))
comb['humid_x_calm']   = comb['humidity_proxy'] * comb['calm_flag']
comb['temp_x_winter']  = comb['temperature'] * comb['is_winter']
comb['o3_x_summer']    = comb['o3'] * (~comb['month'].isin([11,12,1,2,3])).astype(int)
for h in [1,3,6,12,24]:
    comb[f'pressure_d{h}'] = comb['pressure'] - comb['pressure'].shift(h)
    comb[f'temp_d{h}']     = comb['temperature'] - comb['temperature'].shift(h)

# Exogenous lags (shared by both AR and no-AR)
LAG_H  = [1,2,3,6,12,24,48,72,168,336]
ROLL_W = [3,6,12,24,48,168]
EW_H   = [3,6,12,24,72]
EXOG_COLS = BASE + ['pm10_per_wind','co_per_wind','pollution_load','humidity_proxy','atm_stability']

for col in EXOG_COLS:
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

log(f"Exog features built. comb: {comb.shape}")

# ── Split ─────────────────────────────────────────────────────────────────────
trn_e = comb[~comb['_t']].drop(columns=['_t']).reset_index(drop=True)
tst_e = comb[ comb['_t']].drop(columns=['_t']).reset_index(drop=True)
trn_e['pm25_orig'] = train['pm25_orig'].values
trn_e['pm25_log']  = train['pm25_log'].values

# PM25 AR features for training
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

# Test AR cols filled with NaN (filled recursively at inference)
for lag in PM25_LAGS:
    tst_e[f'pm25_lag{lag}'] = np.nan
for w in PM25_ROLL:
    for s in ['mean','std','min','max']:
        tst_e[f'pm25_r{w}_{s}'] = np.nan
for hl in PM25_EW:
    tst_e[f'pm25_ew{hl}'] = np.nan

# Feature sets
EXCL = {'record_id','datetime','pm25','pm25_orig','pm25_log','_t',
        'wind_direction','wind_dir_deg','wind_dir_rad'}
AR_COLS   = [c for c in trn_e.columns if c not in EXCL]                          # with pm25 lags
NOAR_COLS = [c for c in AR_COLS if not c.startswith('pm25_')]                    # no pm25 lags
log(f"AR features: {len(AR_COLS)}  NoAR features: {len(NOAR_COLS)}")

X_ar   = trn_e[AR_COLS].values.astype(np.float32)
X_nar  = trn_e[NOAR_COLS].values.astype(np.float32)
y      = trn_e['pm25_log'].values
yo     = trn_e['pm25_orig'].values
Xt_nar = tst_e[NOAR_COLS].values.astype(np.float32)
Xt_ar  = tst_e[AR_COLS].values.astype(np.float32)    # AR cols = NaN, filled recursively

# Index maps for recursive filling
feat_idx_ar = {col: i for i, col in enumerate(AR_COLS)}
PM25_LAG_IDX  = [(lag, feat_idx_ar[f'pm25_lag{lag}'])  for lag in PM25_LAGS if f'pm25_lag{lag}' in feat_idx_ar]
PM25_ROLL_IDX = [(w, s, feat_idx_ar[f'pm25_r{w}_{s}']) for w in PM25_ROLL for s in ['mean','std','min','max'] if f'pm25_r{w}_{s}' in feat_idx_ar]
PM25_EW_IDX   = [(hl, feat_idx_ar[f'pm25_ew{hl}'])     for hl in PM25_EW   if f'pm25_ew{hl}' in feat_idx_ar]

# ── LGB params ─────────────────────────────────────────────────────────────────
try:
    v2s = optuna.load_study(study_name='lgbm_v2', storage=f'sqlite:///{OUT}/study_v2.db')
    bp  = v2s.best_params
    log(f"Loaded V2 params")
except:
    bp = {}

LGB_P = dict(
    learning_rate     = bp.get('lr', 0.04),
    num_leaves        = int(bp.get('nl', 349)),
    min_child_samples = int(bp.get('mcs', 32)),
    feature_fraction  = bp.get('ff', 0.95),
    bagging_fraction  = bp.get('bf', 0.84),
    reg_alpha         = bp.get('ra', 0.01),
    reg_lambda        = bp.get('rl', 0.16),
    min_split_gain    = bp.get('msg', 0.04),
    n_estimators=2500, bagging_freq=1, n_jobs=-1, random_state=42, verbose=-1
)
HGB_P = dict(max_iter=400, learning_rate=0.03, max_leaf_nodes=127,
             min_samples_leaf=25, l2_regularization=0.5,
             max_depth=8, early_stopping=False, random_state=42)

# ── 3-Fold CV: train AR and No-AR together ────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=3)
oof_ar_lgb = np.full(len(y), np.nan)
oof_ar_hgb = np.full(len(y), np.nan)
oof_nar_lgb = np.full(len(y), np.nan)
oof_nar_hgb = np.full(len(y), np.nan)
ar_lgb_iters, ar_hgb_iters = [], []
nar_lgb_iters = []

log("\n=== 3-Fold CV (AR + No-AR simultaneously) ===")
for fold, (tri, vali) in enumerate(tscv.split(X_ar)):
    ft = time.time()

    # AR LGB
    m = lgb.LGBMRegressor(**LGB_P)
    m.fit(X_ar[tri], y[tri], eval_set=[(X_ar[vali], y[vali])],
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
    oof_ar_lgb[vali] = m.predict(X_ar[vali]); ar_lgb_iters.append(m.best_iteration_)
    r_ar_lgb = rmse_orig(yo[vali], oof_ar_lgb[vali])

    # AR HGB
    mh = HistGradientBoostingRegressor(**HGB_P)
    mh.fit(X_ar[tri], y[tri])
    oof_ar_hgb[vali] = mh.predict(X_ar[vali]); ar_hgb_iters.append(HGB_P['max_iter'])
    r_ar_hgb = rmse_orig(yo[vali], oof_ar_hgb[vali])

    # No-AR LGB
    mn = lgb.LGBMRegressor(**LGB_P)
    mn.fit(X_nar[tri], y[tri], eval_set=[(X_nar[vali], y[vali])],
           callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
    oof_nar_lgb[vali] = mn.predict(X_nar[vali]); nar_lgb_iters.append(mn.best_iteration_)
    r_nar_lgb = rmse_orig(yo[vali], oof_nar_lgb[vali])

    # No-AR HGB
    mnh = HistGradientBoostingRegressor(**HGB_P)
    mnh.fit(X_nar[tri], y[tri])
    oof_nar_hgb[vali] = mnh.predict(X_nar[vali])
    r_nar_hgb = rmse_orig(yo[vali], oof_nar_hgb[vali])

    log(f"  fold {fold+1}: AR-LGB={r_ar_lgb:.4f}(n={m.best_iteration_})  AR-HGB={r_ar_hgb:.4f}  "
        f"NoAR-LGB={r_nar_lgb:.4f}(n={mn.best_iteration_})  NoAR-HGB={r_nar_hgb:.4f}  ({time.time()-ft:.0f}s)")

mask = ~(np.isnan(oof_ar_lgb)|np.isnan(oof_ar_hgb)|np.isnan(oof_nar_lgb)|np.isnan(oof_nar_hgb))
r_ar_lgb_oof  = rmse_orig(yo[mask], oof_ar_lgb[mask])
r_ar_hgb_oof  = rmse_orig(yo[mask], oof_ar_hgb[mask])
r_nar_lgb_oof = rmse_orig(yo[mask], oof_nar_lgb[mask])
r_nar_hgb_oof = rmse_orig(yo[mask], oof_nar_hgb[mask])
log(f"\n  AR-LGB  OOF: {r_ar_lgb_oof:.4f}")
log(f"  AR-HGB  OOF: {r_ar_hgb_oof:.4f}")
log(f"  NoAR-LGB OOF: {r_nar_lgb_oof:.4f}")
log(f"  NoAR-HGB OOF: {r_nar_hgb_oof:.4f}")

# ── Optuna blend over all 4 OOF streams ──────────────────────────────────────
log("\nBlend optimization (400 trials)...")
oof_stack = np.stack([oof_ar_lgb[mask], oof_ar_hgb[mask], oof_nar_lgb[mask], oof_nar_hgb[mask]], axis=1)

def obj(trial):
    w0 = trial.suggest_float('w0',0,1)
    w1 = trial.suggest_float('w1',0,1-w0)
    w2 = trial.suggest_float('w2',0,1-w0-w1)
    w3 = 1-w0-w1-w2
    b  = oof_stack @ np.array([w0,w1,w2,w3])
    return rmse_orig(yo[mask], b)

bs = optuna.create_study(direction='minimize')
bs.optimize(obj, n_trials=400, n_jobs=1)
w0 = bs.best_params['w0']; w1 = bs.best_params['w1']
w2 = bs.best_params['w2']; w3 = 1-w0-w1-w2
r_blend = bs.best_value
log(f"  Blend OOF: {r_blend:.4f}  wAR_L={w0:.3f} wAR_H={w1:.3f} wNAR_L={w2:.3f} wNAR_H={w3:.3f}")

# ── Final models on all training data ─────────────────────────────────────────
log("\nFinal training on all data...")
n_ar_lgb  = max(150, int(np.mean([b for b in ar_lgb_iters  if b and b>0])))
n_nar_lgb = max(150, int(np.mean([b for b in nar_lgb_iters if b and b>0])))

LGB_P['n_estimators'] = n_ar_lgb
lgb_ar_f = lgb.LGBMRegressor(**LGB_P); lgb_ar_f.fit(X_ar, y); log(f"  AR-LGB n={n_ar_lgb}")

hgb_ar_f = HistGradientBoostingRegressor(**HGB_P); hgb_ar_f.fit(X_ar, y); log("  AR-HGB done")

LGB_P['n_estimators'] = n_nar_lgb
lgb_nar_f = lgb.LGBMRegressor(**LGB_P); lgb_nar_f.fit(X_nar, y); log(f"  NoAR-LGB n={n_nar_lgb}")

hgb_nar_f = HistGradientBoostingRegressor(**HGB_P); hgb_nar_f.fit(X_nar, y); log("  NoAR-HGB done")

# ── No-AR test prediction (instant, no recursion) ────────────────────────────
log("\nNo-AR test prediction (direct)...")
p_nar_lgb = lgb_nar_f.predict(Xt_nar)
p_nar_hgb = hgb_nar_f.predict(Xt_nar)
log("  NoAR predictions done")

# ── AR recursive test prediction ─────────────────────────────────────────────
log("\nAR recursive test prediction...")
ewma_alphas = {hl: 1-np.exp(-np.log(2)/hl) for hl in PM25_EW}
ewma_state  = {}
for hl in PM25_EW:
    a = ewma_alphas[hl]; v = float(train['pm25_orig'].iloc[0])
    for val in train['pm25_orig'].values[1:]:
        v = a*float(val) + (1-a)*v
    ewma_state[hl] = v

pm25_buf = deque(train['pm25_orig'].values, maxlen=400)
p_ar_lgb_list = []; p_ar_hgb_list = []
t0 = time.time()

for i in range(len(Xt_ar)):
    row = Xt_ar[i].copy()
    buf = list(pm25_buf)
    for lag, idx in PM25_LAG_IDX:
        row[idx] = buf[-lag] if len(buf) >= lag else np.nan
    for w_r, stat, idx in PM25_ROLL_IDX:
        rec = buf[-w_r:] if len(buf) >= w_r else buf
        if not rec: row[idx] = np.nan; continue
        if   stat=='mean': row[idx] = float(np.mean(rec))
        elif stat=='std':  row[idx] = float(np.std(rec)) if len(rec)>1 else 0.
        elif stat=='min':  row[idx] = float(np.min(rec))
        elif stat=='max':  row[idx] = float(np.max(rec))
    for hl, idx in PM25_EW_IDX:
        row[idx] = ewma_state[hl]

    r2 = row.reshape(1,-1)
    p_lgb = lgb_ar_f.predict(r2)[0]
    p_hgb = hgb_ar_f.predict(r2)[0]
    p_ar_lgb_list.append(p_lgb)
    p_ar_hgb_list.append(p_hgb)

    # Use blended prediction as "true" value for next step's lags
    pp = float(np.expm1(w0*p_lgb + w1*p_hgb).clip(0))
    pm25_buf.append(pp)
    for hl in PM25_EW:
        ewma_state[hl] = ewma_alphas[hl]*pp + (1-ewma_alphas[hl])*ewma_state[hl]

    if (i+1) % 1000 == 0:
        log(f"  {i+1}/{len(Xt_ar)} ({time.time()-t0:.0f}s) last={pp:.1f}")

p_ar_lgb = np.array(p_ar_lgb_list)
p_ar_hgb = np.array(p_ar_hgb_list)
log(f"  AR done in {time.time()-t0:.0f}s")

# ── Combine all 4 predictions ─────────────────────────────────────────────────
plog_final = w0*p_ar_lgb + w1*p_ar_hgb + w2*p_nar_lgb + w3*p_nar_hgb
preds_final = np.expm1(plog_final).clip(0)

# Also save AR-only blend (for potential better LB if local gap is small)
plog_ar_only = (w0*p_ar_lgb + w1*p_ar_hgb) / (w0+w1+1e-9)
preds_ar_only = np.expm1(plog_ar_only).clip(0)

sub = test[['record_id']].copy()
sub['predicted_pm25'] = preds_final
sub.to_csv(SUB_V7, index=False)

sub_ar = test[['record_id']].copy()
sub_ar['predicted_pm25'] = preds_ar_only
sub_ar.to_csv(SUB_AR, index=False)

# Also overwrite the main submission_v5.csv in DL
shutil.copy(SUB_V7, os.path.join(DL, "submission_v5.csv"))

log(f"\n{'='*60}")
log(f"V7 COMPLETE  ({(time.time()-t_start)/60:.1f} min total)")
log(f"  AR-LGB  OOF: {r_ar_lgb_oof:.4f}")
log(f"  AR-HGB  OOF: {r_ar_hgb_oof:.4f}")
log(f"  NoAR-LGB OOF: {r_nar_lgb_oof:.4f}")
log(f"  NoAR-HGB OOF: {r_nar_hgb_oof:.4f}")
log(f"  Blend OOF: {r_blend:.4f}  [{w0:.2f}*AR_LGB + {w1:.2f}*AR_HGB + {w2:.2f}*NAR_LGB + {w3:.2f}*NAR_HGB]")
log(f"  Pred (blend): min={preds_final.min():.1f}  mean={preds_final.mean():.1f}  max={preds_final.max():.1f}")
log(f"  Pred (AR):    min={preds_ar_only.min():.1f}  mean={preds_ar_only.mean():.1f}  max={preds_ar_only.max():.1f}")
log(f"  AR+NoAR blend saved: {SUB_V7}")
log(f"  AR-only saved:       {SUB_AR}")
log(f"{'='*60}")

json.dump(dict(ar_lgb=r_ar_lgb_oof, ar_hgb=r_ar_hgb_oof,
               nar_lgb=r_nar_lgb_oof, nar_hgb=r_nar_hgb_oof,
               blend=r_blend, w0=w0, w1=w1, w2=w2, w3=w3,
               ar_lgb_n=n_ar_lgb, nar_lgb_n=n_nar_lgb),
          open(os.path.join(OUT,"v7_meta.json"),'w'), indent=2)
