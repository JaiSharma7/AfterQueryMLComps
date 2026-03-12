"""
PM2.5 v2 — THE WINNING PIPELINE
Key innovations vs v1:
  1. PM2.5 autoregression: lag + rolling + EWMA features from pm25 in training
  2. Recursive test prediction: build pm25 lags from model predictions row-by-row
  3. CatBoost as 4th ensemble member
  4. Extended features: pressure tendency, atm stability, EWMA on all pollutants
  5. Optuna (50 trials, 3-fold) + stacking meta-learner
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os, json, time
from collections import deque
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT  = r"C:\Users\jaish\OneDrive - The University of Texas at Austin\Documents\AfterQueryMLComps\comp5"
LOG  = os.path.join(OUT, "v2_log.txt")
SUB  = os.path.join(OUT, "submission_v2.csv")

def log(m):
    print(m, flush=True)
    with open(LOG, 'a') as f: f.write(m+'\n')

def rmse_orig(yt, yp_log):
    return np.sqrt(mean_squared_error(yt, np.expm1(yp_log).clip(0)))

# =============================================================================
log("="*60)
log("V2 PIPELINE START")

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(r"C:\Users\jaish\Downloads\train (5).csv", parse_dates=['datetime'])
test  = pd.read_csv(r"C:\Users\jaish\Downloads\test (5).csv",  parse_dates=['datetime'])
train = train.sort_values('datetime').reset_index(drop=True)
test  = test.sort_values('datetime').reset_index(drop=True)
log(f"Train {train.shape}  Test {test.shape}")

train['pm25_original'] = train['pm25'].clip(0)
train['pm25_log']      = np.log1p(train['pm25_original'])

# ── Wind encoding ──────────────────────────────────────────────────────────────
WIND = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,
        'SSE':157.5,'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,
        'WNW':292.5,'NW':315,'NNW':337.5}
for df in [train, test]:
    df['wind_dir_deg'] = df['wind_direction'].map(WIND)
    df['wind_dir_rad'] = np.radians(df['wind_dir_deg'])
    df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
    df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])

# ── Temporal features ──────────────────────────────────────────────────────────
for df in [train, test]:
    df['hour_sin']  = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']  = np.cos(2*np.pi*df['hour']/24)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    dow = df['datetime'].dt.dayofweek
    df['dow_sin']  = np.sin(2*np.pi*dow/7)
    df['dow_cos']  = np.cos(2*np.pi*dow/7)
    doy = df['datetime'].dt.dayofyear
    df['doy_sin']  = np.sin(2*np.pi*doy/365)
    df['doy_cos']  = np.cos(2*np.pi*doy/365)
    df['is_weekend']   = (dow >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
    df['is_winter']    = df['month'].isin([11,12,1,2,3]).astype(int)
    df['is_heating_season'] = df['month'].isin([11,12,1,2]).astype(int)

log("Wind + temporal done")

# ── Combined exogenous features (NO pm25) ──────────────────────────────────────
BASE = ['pm10','so2','no2','co','o3','temperature','pressure','dew_point','wind_speed','rain']

tc = train.drop(columns=['pm25','pm25_original','pm25_log'])
tc['is_test'] = False
tec = test.copy(); tec['is_test'] = True
comb = pd.concat([tc, tec], ignore_index=True).sort_values('datetime').reset_index(drop=True)
assert 'pm25' not in comb.columns

# Impute base features
log("Imputing base features...")
for col in BASE + ['wind_dir_deg','wind_dir_rad','wind_dir_sin','wind_dir_cos']:
    comb[col] = comb[col].interpolate(method='linear', limit=6, limit_direction='both').ffill().bfill()

# Derived meteorological features
comb['humidity_proxy']      = comb['dew_point'] - comb['temperature']
comb['wind_u']              = comb['wind_speed'] * np.sin(comb['wind_dir_rad'])
comb['wind_v']              = comb['wind_speed'] * np.cos(comb['wind_dir_rad'])
comb['atm_stability_proxy'] = comb['temperature'] - comb['dew_point']  # dew pt depression; low = stable = pollution trapping
comb['pressure_1h_change']  = comb['pressure'] - comb['pressure'].shift(1)
comb['pressure_3h_change']  = comb['pressure'] - comb['pressure'].shift(3)
comb['pressure_6h_change']  = comb['pressure'] - comb['pressure'].shift(6)
comb['calm_flag']           = (comb['wind_speed'] < 1.0).astype(int)

log("Building exogenous lag/rolling/EWMA features...")

# Lags for exogenous (extended: add 336h = 2 weeks)
LAG_H = [1,2,3,6,12,24,48,72,168,336]
for col in BASE:
    for lag in LAG_H:
        comb[f'{col}_lag_{lag}h'] = comb[col].shift(lag)

# Rolling for exogenous
ROLL_W = [3,6,12,24,48,168]
for col in BASE:
    for w in ROLL_W:
        r = comb[col].rolling(w, min_periods=1)
        comb[f'{col}_roll_{w}h_mean'] = r.mean()
        comb[f'{col}_roll_{w}h_std']  = r.std()
        comb[f'{col}_roll_{w}h_min']  = r.min()
        comb[f'{col}_roll_{w}h_max']  = r.max()

# EWMA for exogenous
EWMA_HL = [3, 6, 12, 24, 72]
for col in BASE:
    for hl in EWMA_HL:
        comb[f'{col}_ewma_{hl}h'] = comb[col].ewm(halflife=hl, min_periods=1).mean()

# Interaction features
comb['pollution_accum_proxy'] = comb['pm10_roll_24h_mean'] / (comb['wind_speed'].clip(lower=0.1))
comb['pressure_delta_24h']    = comb['pressure'] - comb['pressure_lag_24h']

log(f"Combined exog shape: {comb.shape}")

# Split back
trn_exog = comb[~comb['is_test']].drop(columns=['is_test']).reset_index(drop=True)
tst_exog = comb[ comb['is_test']].drop(columns=['is_test']).reset_index(drop=True)

# =============================================================================
# PM2.5 AUTOREGRESSIVE FEATURES (training only — actual values)
# These are the KEY features missing from v1
# =============================================================================
log("Building PM2.5 autoregressive features on training set...")

PM25_LAGS = [1,2,3,6,12,24,48,72,168]
PM25_ROLL_W = [3,6,12,24,48,168]
PM25_EWMA_HL = [3,6,12,24,72]

# Work on training portion only (using actual pm25)
pm25_series = train['pm25_original'].copy()

for lag in PM25_LAGS:
    trn_exog[f'pm25_lag_{lag}h'] = pm25_series.shift(lag).values

for w in PM25_ROLL_W:
    r = pm25_series.shift(1).rolling(w, min_periods=1)  # shift(1) prevents current-value leakage
    trn_exog[f'pm25_roll_{w}h_mean'] = r.mean().values
    trn_exog[f'pm25_roll_{w}h_std']  = r.std().values
    trn_exog[f'pm25_roll_{w}h_min']  = r.min().values
    trn_exog[f'pm25_roll_{w}h_max']  = r.max().values

for hl in PM25_EWMA_HL:
    trn_exog[f'pm25_ewma_{hl}h'] = pm25_series.shift(1).ewm(halflife=hl, min_periods=1).mean().values

# Add targets
trn_exog['pm25_original'] = train['pm25_original'].values
trn_exog['pm25_log']      = train['pm25_log'].values

# For test, pm25 features are NaN for now (filled recursively during prediction)
for lag in PM25_LAGS:
    tst_exog[f'pm25_lag_{lag}h'] = np.nan
for w in PM25_ROLL_W:
    for stat in ['mean','std','min','max']:
        tst_exog[f'pm25_roll_{w}h_{stat}'] = np.nan
for hl in PM25_EWMA_HL:
    tst_exog[f'pm25_ewma_{hl}h'] = np.nan

log(f"train_feat: {trn_exog.shape}  test_feat: {tst_exog.shape}")

# =============================================================================
# Feature selection
# =============================================================================
EXCL = {'record_id','datetime','pm25','pm25_original','pm25_log',
        'is_test','wind_direction','wind_dir_deg','wind_dir_rad'}
FCOLS = [c for c in trn_exog.columns if c not in EXCL]
log(f"Total features: {len(FCOLS)}")

X  = trn_exog[FCOLS].values.astype(np.float32)
y  = trn_exog['pm25_log'].values
yo = trn_exog['pm25_original'].values

# Build X_test base matrix (pm25 features = NaN, will be filled recursively)
X_test_base = tst_exog[FCOLS].values.astype(np.float32)

# Precompute index lookup for pm25 feature columns
feat_idx = {col: i for i, col in enumerate(FCOLS)}
PM25_LAG_IDX   = [(lag, feat_idx[f'pm25_lag_{lag}h']) for lag in PM25_LAGS if f'pm25_lag_{lag}h' in feat_idx]
PM25_ROLL_IDX  = [(w, stat, feat_idx[f'pm25_roll_{w}h_{stat}']) for w in PM25_ROLL_W for stat in ['mean','std','min','max'] if f'pm25_roll_{w}h_{stat}' in feat_idx]
PM25_EWMA_IDX  = [(hl, feat_idx[f'pm25_ewma_{hl}h']) for hl in PM25_EWMA_HL if f'pm25_ewma_{hl}h' in feat_idx]

tscv  = TimeSeriesSplit(n_splits=5)
tscv3 = TimeSeriesSplit(n_splits=3)  # for Optuna (faster)

# =============================================================================
# CV helper
# =============================================================================
def cv_oof(model_fn, name, cv=None):
    if cv is None: cv = tscv
    oof = np.full(len(X), np.nan)
    rmses, best_iters = [], []
    for fold, (tri, vali) in enumerate(cv.split(X)):
        m = model_fn()
        Xtr, Xv = X[tri], X[vali]
        ytr, yv = y[tri], y[vali]
        if isinstance(m, lgb.LGBMRegressor):
            m.fit(Xtr, ytr, eval_set=[(Xv, yv)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
            bi = m.best_iteration_
        elif isinstance(m, xgb.XGBRegressor):
            m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
            bi = m.best_iteration
        elif isinstance(m, cb.CatBoostRegressor):
            m.fit(Xtr, ytr, eval_set=(Xv, yv), verbose=False)
            bi = m.get_best_iteration()
        else:
            m.fit(Xtr, ytr); bi = 0
        pred = m.predict(Xv)
        oof[vali] = pred
        r = rmse_orig(yo[vali], pred)
        rmses.append(r)
        best_iters.append(bi or 0)
        log(f"    {name} fold {fold+1}: RMSE={r:.4f}  iter={bi}")
    mask = ~np.isnan(oof)
    oof_r = rmse_orig(yo[mask], oof[mask])
    vi = [b for b in best_iters if b and b > 0]
    log(f"  {name} OOF RMSE: {oof_r:.4f}  avg_iter={int(np.mean(vi)) if vi else 0}")
    return oof, oof_r, best_iters

# =============================================================================
# STEP 9: LightGBM
# =============================================================================
log("\nSTEP 9: LightGBM CV")
t0=time.time()
lgbm_p = dict(n_estimators=3000, learning_rate=0.05, num_leaves=127,
              min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, reg_alpha=0.1, reg_lambda=1.0,
              n_jobs=-1, random_state=42, verbose=-1)
oof_lgb, rmse_lgb, lgb_iters = cv_oof(lambda: lgb.LGBMRegressor(**lgbm_p), "LGB")
log(f"  done {time.time()-t0:.0f}s")

# =============================================================================
# STEP 10: XGBoost
# =============================================================================
log("\nSTEP 10: XGBoost CV")
t0=time.time()
n_xgb = max(200, int(np.mean([b for b in lgb_iters if b>0])+100))
xgb_p = dict(n_estimators=n_xgb, learning_rate=0.05, max_depth=6,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
             reg_alpha=0.1, reg_lambda=1.0, tree_method='hist',
             early_stopping_rounds=100, n_jobs=-1, random_state=42, verbosity=0)
oof_xgb, rmse_xgb, xgb_iters = cv_oof(lambda: xgb.XGBRegressor(**xgb_p), "XGB")
log(f"  done {time.time()-t0:.0f}s")

# =============================================================================
# STEP 11: HGBR
# =============================================================================
log("\nSTEP 11: HGBR CV")
t0=time.time()
n_hgb = max(200, int(np.mean([b for b in lgb_iters if b>0])))
hgb_p = dict(max_iter=n_hgb, learning_rate=0.05, max_leaf_nodes=127,
             min_samples_leaf=20, l2_regularization=1.0,
             early_stopping=False, random_state=42)
oof_hgb, rmse_hgb, _ = cv_oof(lambda: HistGradientBoostingRegressor(**hgb_p), "HGB")
log(f"  done {time.time()-t0:.0f}s")

# =============================================================================
# STEP 11b: CatBoost
# =============================================================================
log("\nSTEP 11b: CatBoost CV")
t0=time.time()
n_cat = max(200, int(np.mean([b for b in lgb_iters if b>0])))
cat_p = dict(iterations=n_cat, learning_rate=0.05, depth=6,
             l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.8,
             eval_metric='RMSE', random_seed=42, thread_count=-1,
             early_stopping_rounds=100, verbose=False)
oof_cat, rmse_cat, cat_iters = cv_oof(lambda: cb.CatBoostRegressor(**cat_p), "CAT")
log(f"  done {time.time()-t0:.0f}s")

log(f"\nBaseline summary: LGB={rmse_lgb:.4f}  XGB={rmse_xgb:.4f}  HGB={rmse_hgb:.4f}  CAT={rmse_cat:.4f}")

# =============================================================================
# STEP 12: Optuna LightGBM tuning (50 trials, 3-fold)
# =============================================================================
log("\nSTEP 12: Optuna LightGBM (50 trials, 3-fold)...")
t0=time.time()

def lgbm_obj(trial):
    p = dict(
        n_estimators=max(200, int(np.mean([b for b in lgb_iters if b>0]))+200),
        learning_rate=trial.suggest_float('lr', 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int('nl', 63, 511),
        min_child_samples=trial.suggest_int('mcs', 5, 80),
        feature_fraction=trial.suggest_float('ff', 0.5, 1.0),
        bagging_fraction=trial.suggest_float('bf', 0.5, 1.0),
        bagging_freq=1,
        reg_alpha=trial.suggest_float('ra', 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float('rl', 1e-4, 10.0, log=True),
        min_split_gain=trial.suggest_float('msg', 0, 1.0),
        n_jobs=-1, random_state=42, verbose=-1
    )
    rs = []
    for tri, vali in tscv3.split(X):
        m = lgb.LGBMRegressor(**p)
        m.fit(X[tri], y[tri], eval_set=[(X[vali], y[vali])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        rs.append(rmse_orig(yo[vali], m.predict(X[vali])))
    return np.mean(rs)

study = optuna.create_study(direction='minimize',
    storage=f'sqlite:///{OUT}/study_v2.db', study_name='lgbm_v2',
    load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=8))
study.optimize(lgbm_obj, n_trials=50)
log(f"  Optuna best: {study.best_value:.4f}  ({time.time()-t0:.0f}s)")
log(f"  Params: {study.best_params}")

# Re-run OOF with tuned LGB
best_p = study.best_params.copy()
best_p.update({'n_estimators': max(200, int(np.mean([b for b in lgb_iters if b>0]))+200),
               'bagging_freq':1, 'n_jobs':-1, 'random_state':42, 'verbose':-1})
oof_lgb_t, rmse_lgb_t, lgb_iters_t = cv_oof(lambda: lgb.LGBMRegressor(**best_p), "LGB_tuned")
if rmse_lgb_t < rmse_lgb:
    oof_lgb = oof_lgb_t; rmse_lgb = rmse_lgb_t; lgb_iters = lgb_iters_t; lgbm_p = best_p
    log("  Using TUNED LGB")
else:
    log("  Keeping DEFAULT LGB")

# =============================================================================
# STEP 12b: Stacking meta-learner
# =============================================================================
log("\nSTEP 12b: Stacking meta-learner...")
mask = ~(np.isnan(oof_lgb)|np.isnan(oof_xgb)|np.isnan(oof_hgb)|np.isnan(oof_cat))
S = np.column_stack([oof_lgb[mask], oof_xgb[mask], oof_hgb[mask], oof_cat[mask]])
meta = Ridge(alpha=1.0, positive=True)  # positive weights (no negative blending)
meta.fit(S, yo[mask])  # fit on original scale? No — fit on log predictions vs log targets
# Fit meta on log predictions vs log targets
meta_log = Ridge(alpha=1.0, positive=True)
meta_log.fit(S, y[mask])
oof_stack_log = meta_log.predict(S)
rmse_stack = rmse_orig(yo[mask], oof_stack_log)
log(f"  Stack OOF RMSE: {rmse_stack:.4f}  weights={meta_log.coef_.round(3)}")

# Simple equal blend for comparison
rmse_eq = rmse_orig(yo[mask], ((oof_lgb+oof_xgb+oof_hgb+oof_cat)/4)[mask])
log(f"  Equal blend RMSE: {rmse_eq:.4f}")

# Optuna blend weights
def blend_obj(trial):
    w = np.array([trial.suggest_float(f'w{i}',0,1) for i in range(4)])
    w = w/w.sum()
    b = w[0]*oof_lgb[mask]+w[1]*oof_xgb[mask]+w[2]*oof_hgb[mask]+w[3]*oof_cat[mask]
    return rmse_orig(yo[mask], b)

bs = optuna.create_study(direction='minimize')
bs.optimize(blend_obj, n_trials=500)
bw = np.array([bs.best_params[f'w{i}'] for i in range(4)])
bw /= bw.sum()
rmse_blend = bs.best_value
log(f"  Optuna blend RMSE: {rmse_blend:.4f}  w={bw.round(3)}")

# Pick best ensemble strategy
best_rmse = min(rmse_stack, rmse_blend, rmse_eq)
if rmse_stack == best_rmse:
    log("  Using STACKING")
    use_stack = True; use_blend = False
elif rmse_blend < rmse_eq - 0.3:
    log("  Using OPTUNA BLEND weights")
    use_stack = False; use_blend = True; w_final = bw
else:
    log("  Using EQUAL blend")
    use_stack = False; use_blend = False; w_final = np.array([0.25,0.25,0.25,0.25])

# =============================================================================
# STEP 13: Train final models on full data
# =============================================================================
log("\nSTEP 13: Final model training on full data...")

n_lgb_final = max(200, int(np.mean([b for b in lgb_iters if b and b>0])))
lgbm_p['n_estimators'] = n_lgb_final
lgb_final = lgb.LGBMRegressor(**lgbm_p)
lgb_final.fit(X, y)
log(f"  LGB trained n={n_lgb_final}")

n_xgb_final = max(200, int(np.mean([b for b in xgb_iters if b and b>0]) or n_lgb_final))
xgb_p2 = {k:v for k,v in xgb_p.items() if k!='early_stopping_rounds'}
xgb_p2['n_estimators'] = n_xgb_final
xgb_final = xgb.XGBRegressor(**xgb_p2)
xgb_final.fit(X, y)
log(f"  XGB trained n={n_xgb_final}")

hgb_p['max_iter'] = n_lgb_final
hgb_final = HistGradientBoostingRegressor(**hgb_p)
hgb_final.fit(X, y)
log("  HGB trained")

n_cat_final = max(200, int(np.mean([b for b in cat_iters if b and b>0]) or n_lgb_final))
cat_p2 = {k:v for k,v in cat_p.items() if k not in ['early_stopping_rounds']}
cat_p2['iterations'] = n_cat_final
cat_final = cb.CatBoostRegressor(**cat_p2)
cat_final.fit(X, y, verbose=False)
log(f"  CAT trained n={n_cat_final}")

if use_stack:
    log("  Re-fitting meta-learner on full data OOF...")
    # Note: for full-data stacking we use the OOF predictions (already have them)
    meta_log.fit(np.column_stack([oof_lgb[mask],oof_xgb[mask],oof_hgb[mask],oof_cat[mask]]),
                 y[mask])

# =============================================================================
# STEP 14: RECURSIVE TEST PREDICTION
# The KEY innovation: use predicted pm25 as lag/rolling/EWMA features
# =============================================================================
log("\nSTEP 14: Recursive test prediction (row by row)...")

# Initialize pm25 buffer from training tail (need at least 336 values)
pm25_buf = deque(train['pm25_original'].values, maxlen=400)

# Initialize EWMA states from end of training
ewma_alphas = {hl: 1 - np.exp(-np.log(2)/hl) for hl in PM25_EWMA_HL}
ewma_state = {}
for hl in PM25_EWMA_HL:
    alpha = ewma_alphas[hl]
    v = float(train['pm25_original'].iloc[0])
    for val in train['pm25_original'].values[1:]:
        v = alpha*float(val) + (1-alpha)*v
    ewma_state[hl] = v

test_preds_log = []

def predict_ensemble(row_2d):
    """Predict from a single row (1, n_features)"""
    p_lgb = lgb_final.predict(row_2d)[0]
    p_xgb = xgb_final.predict(row_2d)[0]
    p_hgb = hgb_final.predict(row_2d)[0]
    p_cat = cat_final.predict(row_2d)[0]
    if use_stack:
        return meta_log.predict(np.array([[p_lgb,p_xgb,p_hgb,p_cat]]))[0]
    elif use_blend:
        return w_final[0]*p_lgb + w_final[1]*p_xgb + w_final[2]*p_hgb + w_final[3]*p_cat
    else:
        return (p_lgb+p_xgb+p_hgb+p_cat)/4

t0 = time.time()
for i in range(len(X_test_base)):
    row = X_test_base[i].copy()
    buf = list(pm25_buf)

    # Fill pm25 lag features
    for lag, idx in PM25_LAG_IDX:
        row[idx] = buf[-lag] if len(buf) >= lag else np.nan

    # Fill pm25 rolling features
    for w_r, stat, idx in PM25_ROLL_IDX:
        recent = buf[-w_r:] if len(buf) >= w_r else buf
        if not recent:
            row[idx] = np.nan; continue
        if stat == 'mean': row[idx] = np.mean(recent)
        elif stat == 'std': row[idx] = np.std(recent) if len(recent)>1 else 0.0
        elif stat == 'min': row[idx] = np.min(recent)
        elif stat == 'max': row[idx] = np.max(recent)

    # Fill pm25 EWMA features
    for hl, idx in PM25_EWMA_IDX:
        row[idx] = ewma_state[hl]

    # Predict
    pred_log = predict_ensemble(row.reshape(1,-1))
    test_preds_log.append(pred_log)
    pred_pm25 = float(np.expm1(pred_log).clip(0))

    # Update buffer and EWMA
    pm25_buf.append(pred_pm25)
    for hl in PM25_EWMA_HL:
        a = ewma_alphas[hl]
        ewma_state[hl] = a*pred_pm25 + (1-a)*ewma_state[hl]

    if (i+1) % 1000 == 0:
        log(f"    Predicted {i+1}/{len(X_test_base)} rows  last_pred={pred_pm25:.1f}")

log(f"  Recursive prediction done in {time.time()-t0:.0f}s")

# =============================================================================
# Save submission
# =============================================================================
final_pred = np.expm1(np.array(test_preds_log)).clip(0)
assert len(final_pred)==6828 and not np.any(np.isnan(final_pred))

sub = pd.DataFrame({'record_id': tst_exog['record_id'].values,
                    'predicted_pm25': final_pred})
sub.sort_values('record_id').to_csv(SUB, index=False)

# Also copy to Downloads for easy upload
import shutil
shutil.copy(SUB, r"C:\Users\jaish\Downloads\submission_v2.csv")

log(f"\nSaved: {SUB}")
log(f"Pred stats: min={final_pred.min():.1f}  mean={final_pred.mean():.2f}  max={final_pred.max():.1f}")
log("\n" + "="*60)
log("V2 COMPLETE")
log(f"  LGB OOF RMSE:   {rmse_lgb:.4f}")
log(f"  XGB OOF RMSE:   {rmse_xgb:.4f}")
log(f"  HGB OOF RMSE:   {rmse_hgb:.4f}")
log(f"  CAT OOF RMSE:   {rmse_cat:.4f}")
log(f"  Ensemble RMSE:  {best_rmse:.4f}")
log("="*60)

with open(os.path.join(OUT,'v2_meta.json'),'w') as f:
    json.dump({'lgb':rmse_lgb,'xgb':rmse_xgb,'hgb':rmse_hgb,'cat':rmse_cat,
               'ensemble':best_rmse,'n_features':len(FCOLS),'n_test':len(final_pred)},f,indent=2)
