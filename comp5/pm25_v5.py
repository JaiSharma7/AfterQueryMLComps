"""
V5: LGB (V2-params) + XGB + HGB — NO CatBoost. Fast. 492 features.
Improvements: interaction features, cross-pollutant features, better XGB params.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os, time, json, shutil
from collections import deque
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT   = r"C:\Users\jaish\OneDrive - The University of Texas at Austin\Documents\AfterQueryMLComps\comp5"
DL    = r"C:\Users\jaish\Downloads"
LOG   = os.path.join(OUT, "v5_log.txt")
META  = os.path.join(OUT, "v5_meta.json")
SUB   = os.path.join(OUT, "submission_v5.csv")
DL_SUB= os.path.join(DL,  "submission_v5.csv")

open(LOG, 'w').close()
def log(m): print(m, flush=True); open(LOG,'a').write(m+'\n')

def rmse(yt, yp_log):
    return float(np.sqrt(mean_squared_error(yt, np.expm1(yp_log).clip(0))))

# ── Data ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(DL, "train (5).csv"), parse_dates=['datetime'])
test  = pd.read_csv(os.path.join(DL, "test (5).csv"),  parse_dates=['datetime'])
train = train.sort_values('datetime').reset_index(drop=True)
test  = test.sort_values('datetime').reset_index(drop=True)
train['pm25_orig'] = train['pm25'].clip(lower=0)
train['pm25_log']  = np.log1p(train['pm25_orig'])
log(f"V5: LGB+XGB+HGB | Train={len(train)} Test={len(test)}")

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
    df['dow_sin']  = np.sin(2*np.pi*dow/7)
    df['dow_cos']  = np.cos(2*np.pi*dow/7)
    doy = df['datetime'].dt.dayofyear
    df['doy_sin']  = np.sin(2*np.pi*doy/365)
    df['doy_cos']  = np.cos(2*np.pi*doy/365)
    df['is_weekend']        = (dow >= 5).astype(np.int8)
    df['is_rush_hour']      = df['hour'].isin([7,8,9,17,18,19]).astype(np.int8)
    df['is_winter']         = df['month'].isin([11,12,1,2,3]).astype(np.int8)
    df['is_heating_season'] = df['month'].isin([11,12,1,2]).astype(np.int8)
    df['quarter']           = df['datetime'].dt.quarter
    return df

train = add_temporal(train)
test  = add_temporal(test)

# ── Combined exog interpolation ───────────────────────────────────────────────
BASE = ['pm10','so2','no2','co','o3','temperature','pressure','dew_point','wind_speed','rain']
tc   = train.drop(columns=['pm25','pm25_orig','pm25_log']); tc['_t'] = False
tec  = test.copy(); tec['_t'] = True
comb = pd.concat([tc, tec], ignore_index=True).sort_values('datetime').reset_index(drop=True)

for col in BASE + ['wind_dir_deg','wind_dir_rad','wind_dir_sin','wind_dir_cos']:
    comb[col] = comb[col].interpolate(method='linear', limit=6, limit_direction='both').ffill().bfill()

# Interactions
comb['humidity_proxy']  = comb['dew_point'] - comb['temperature']
comb['wind_u']          = comb['wind_speed'] * np.sin(comb['wind_dir_rad'])
comb['wind_v']          = comb['wind_speed'] * np.cos(comb['wind_dir_rad'])
comb['atm_stability']   = comb['temperature'] - comb['dew_point']
comb['pressure_1h']     = comb['pressure'] - comb['pressure'].shift(1)
comb['pressure_3h']     = comb['pressure'] - comb['pressure'].shift(3)
comb['pressure_6h']     = comb['pressure'] - comb['pressure'].shift(6)
comb['pressure_24h']    = comb['pressure'] - comb['pressure'].shift(24)
comb['calm_flag']       = (comb['wind_speed'] < 1.0).astype(np.int8)
comb['very_calm']       = (comb['wind_speed'] < 0.3).astype(np.int8)
comb['pm10_x_no2']      = comb['pm10'] * comb['no2']
comb['no2_x_co']        = comb['no2'] * comb['co']
comb['so2_x_co']        = comb['so2'] * comb['co']
comb['pollution_load']  = comb['pm10'] + comb['no2'] + comb['so2']
comb['pm10_per_wind']   = comb['pm10'] / (comb['wind_speed'].clip(lower=0.1))
comb['co_per_wind']     = comb['co']  / (comb['wind_speed'].clip(lower=0.1))
comb['humid_x_calm']    = comb['humidity_proxy'] * comb['calm_flag']
comb['temp_x_winter']   = comb['temperature'] * comb['is_winter']
comb['o3_x_summer']     = comb['o3'] * (~comb['month'].isin([11,12,1,2,3])).astype(int)

# Lags + Rolling + EWMA
LAG_HOURS = [1,2,3,6,12,24,48,72,168,336]
ROLL_WINS = [3,6,12,24,48,168]
EW_HALVES = [3,6,12,24,72]

for col in BASE:
    for lag in LAG_HOURS:
        comb[f'{col}_lag{lag}'] = comb[col].shift(lag)
    for w in ROLL_WINS:
        r = comb[col].rolling(w, min_periods=1)
        comb[f'{col}_r{w}_mean'] = r.mean()
        comb[f'{col}_r{w}_std']  = r.std()
        comb[f'{col}_r{w}_min']  = r.min()
        comb[f'{col}_r{w}_max']  = r.max()
    for hl in EW_HALVES:
        comb[f'{col}_ew{hl}'] = comb[col].ewm(halflife=hl, min_periods=1).mean()

for col in ['pm10_per_wind','humidity_proxy','pollution_load']:
    for lag in [1,3,6,12,24]:
        comb[f'{col}_lag{lag}'] = comb[col].shift(lag)

log(f"Features built. comb shape: {comb.shape}")

# ── Split ─────────────────────────────────────────────────────────────────────
trn_exog = comb[~comb['_t']].drop(columns=['_t']).reset_index(drop=True)
tst_exog = comb[ comb['_t']].drop(columns=['_t']).reset_index(drop=True)
trn_exog['pm25_orig'] = train['pm25_orig'].values
trn_exog['pm25_log']  = train['pm25_log'].values

pm25_s = train['pm25_orig'].copy()
PM25_LAGS = [1,2,3,6,12,24,48,72,168,336]
PM25_ROLL = [3,6,12,24,48,168]
PM25_EW   = [3,6,12,24,72]

for lag in PM25_LAGS:
    trn_exog[f'pm25_lag{lag}'] = pm25_s.shift(lag).values
for w in PM25_ROLL:
    r = pm25_s.shift(1).rolling(w, min_periods=1)
    trn_exog[f'pm25_r{w}_mean'] = r.mean().values
    trn_exog[f'pm25_r{w}_std']  = r.std().values
    trn_exog[f'pm25_r{w}_min']  = r.min().values
    trn_exog[f'pm25_r{w}_max']  = r.max().values
for hl in PM25_EW:
    trn_exog[f'pm25_ew{hl}'] = pm25_s.shift(1).ewm(halflife=hl, min_periods=1).mean().values

for lag in PM25_LAGS:
    tst_exog[f'pm25_lag{lag}'] = np.nan
for w in PM25_ROLL:
    for stat in ['mean','std','min','max']:
        tst_exog[f'pm25_r{w}_{stat}'] = np.nan
for hl in PM25_EW:
    tst_exog[f'pm25_ew{hl}'] = np.nan

EXCL = {'record_id','datetime','pm25','pm25_orig','pm25_log','_t',
        'wind_direction','wind_dir_deg','wind_dir_rad'}
FCOLS = [c for c in trn_exog.columns if c not in EXCL]
log(f"Features: {len(FCOLS)}")

X  = trn_exog[FCOLS].values.astype(np.float32)
y  = trn_exog['pm25_log'].values
yo = trn_exog['pm25_orig'].values
X_test_base = tst_exog[FCOLS].values.astype(np.float32)

feat_idx      = {col: i for i, col in enumerate(FCOLS)}
PM25_LAG_IDX  = [(lag, feat_idx[f'pm25_lag{lag}'])  for lag in PM25_LAGS if f'pm25_lag{lag}' in feat_idx]
PM25_ROLL_IDX = [(w, s, feat_idx[f'pm25_r{w}_{s}']) for w in PM25_ROLL for s in ['mean','std','min','max'] if f'pm25_r{w}_{s}' in feat_idx]
PM25_EW_IDX   = [(hl, feat_idx[f'pm25_ew{hl}'])     for hl in PM25_EW   if f'pm25_ew{hl}' in feat_idx]

# ── Load V2 best LGB params ───────────────────────────────────────────────────
try:
    v2s = optuna.load_study(study_name='lgbm_v2', storage=f'sqlite:///{OUT}/study_v2.db')
    bp  = v2s.best_params
    log(f"Loaded V2 LGB params (trial RMSE={v2s.best_value:.4f})")
except:
    bp = dict(lr=0.04, nl=349, mcs=32, ff=0.95, bf=0.84, ra=0.01, rl=0.16, msg=0.04)

lgb_p = dict(
    learning_rate     = bp.get('lr', bp.get('learning_rate', 0.04)),
    num_leaves        = int(bp.get('nl', bp.get('num_leaves', 349))),
    min_child_samples = int(bp.get('mcs', bp.get('min_child_samples', 32))),
    feature_fraction  = bp.get('ff', bp.get('feature_fraction', 0.95)),
    bagging_fraction  = bp.get('bf', bp.get('bagging_fraction', 0.84)),
    reg_alpha         = bp.get('ra', bp.get('reg_alpha', 0.01)),
    reg_lambda        = bp.get('rl', bp.get('reg_lambda', 0.16)),
    min_split_gain    = bp.get('msg', bp.get('min_split_gain', 0.04)),
    n_estimators=3000, bagging_freq=1, n_jobs=-1, random_state=42, verbose=-1
)
hgb_p = dict(max_iter=500, learning_rate=0.025, max_leaf_nodes=127,
             min_samples_leaf=25, l2_regularization=0.5,
             max_depth=8, early_stopping=False, random_state=42)
xgb_p = dict(n_estimators=3000, learning_rate=0.04, max_depth=7,
             min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
             reg_alpha=0.05, reg_lambda=1.0, tree_method='hist',
             n_jobs=-1, random_state=42, verbosity=0, early_stopping_rounds=100)

# ── 5-Fold CV ─────────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
oof_lgb = np.full(len(X), np.nan)
oof_hgb = np.full(len(X), np.nan)
oof_xgb = np.full(len(X), np.nan)
lgb_iters, xgb_iters = [], []
t_cv = time.time()

log("\nSTEP 1: 5-fold CV")
for fold, (tri, vali) in enumerate(tscv.split(X)):
    ft = time.time()
    m_lgb = lgb.LGBMRegressor(**lgb_p)
    m_lgb.fit(X[tri], y[tri], eval_set=[(X[vali], y[vali])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb[vali] = m_lgb.predict(X[vali])
    lgb_iters.append(m_lgb.best_iteration_)
    r_lgb = rmse(yo[vali], oof_lgb[vali])

    m_xgb = xgb.XGBRegressor(**xgb_p)
    m_xgb.fit(X[tri], y[tri], eval_set=[(X[vali], y[vali])], verbose=False)
    oof_xgb[vali] = m_xgb.predict(X[vali])
    xgb_iters.append(m_xgb.best_iteration)
    r_xgb = rmse(yo[vali], oof_xgb[vali])

    m_hgb = HistGradientBoostingRegressor(**hgb_p)
    m_hgb.fit(X[tri], y[tri])
    oof_hgb[vali] = m_hgb.predict(X[vali])
    r_hgb = rmse(yo[vali], oof_hgb[vali])

    log(f"  fold {fold+1}: LGB={r_lgb:.4f}(n={m_lgb.best_iteration_})  "
        f"XGB={r_xgb:.4f}(n={m_xgb.best_iteration})  HGB={r_hgb:.4f}  ({time.time()-ft:.0f}s)")

mask = ~(np.isnan(oof_lgb)|np.isnan(oof_xgb)|np.isnan(oof_hgb))
r_lgb_oof = rmse(yo[mask], oof_lgb[mask])
r_xgb_oof = rmse(yo[mask], oof_xgb[mask])
r_hgb_oof = rmse(yo[mask], oof_hgb[mask])
log(f"\n  LGB OOF: {r_lgb_oof:.4f}  XGB OOF: {r_xgb_oof:.4f}  HGB OOF: {r_hgb_oof:.4f}")
log(f"  CV total: {(time.time()-t_cv)/60:.1f}min")

# ── Optuna blend ──────────────────────────────────────────────────────────────
log("\nSTEP 2: Optuna blend (600 trials) ...")
def obj(trial):
    wl = trial.suggest_float('wl', 0.0, 1.0)
    wx = trial.suggest_float('wx', 0.0, 1.0 - wl)
    wh = 1.0 - wl - wx
    b  = wl*oof_lgb[mask] + wx*oof_xgb[mask] + wh*oof_hgb[mask]
    return rmse(yo[mask], b)

bs = optuna.create_study(direction='minimize')
bs.optimize(obj, n_trials=600, n_jobs=1)
wl = bs.best_params['wl']
wx = bs.best_params['wx']
wh = 1.0 - wl - wx
r_blend = bs.best_value
log(f"  Blend RMSE: {r_blend:.4f}  wLGB={wl:.3f}  wXGB={wx:.3f}  wHGB={wh:.3f}")

# ── Final models on all data ──────────────────────────────────────────────────
log("\nSTEP 3: Final models on all training data ...")
n_lgb = max(200, int(np.mean([b for b in lgb_iters if b and b > 0])))
n_xgb = max(200, int(np.mean([b for b in xgb_iters if b and b > 0])))

lgb_p['n_estimators'] = n_lgb
lgb_final = lgb.LGBMRegressor(**lgb_p)
lgb_final.fit(X, y)
log(f"  LGB n={n_lgb}")

xgb_p_final = {k: v for k, v in xgb_p.items() if k != 'early_stopping_rounds'}
xgb_p_final['n_estimators'] = n_xgb
xgb_final = xgb.XGBRegressor(**xgb_p_final)
xgb_final.fit(X, y)
log(f"  XGB n={n_xgb}")

hgb_final = HistGradientBoostingRegressor(**hgb_p)
hgb_final.fit(X, y)
log("  HGB done")

# ── Recursive test prediction ─────────────────────────────────────────────────
log("\nSTEP 4: Recursive test prediction ...")
ewma_alphas = {hl: 1 - np.exp(-np.log(2)/hl) for hl in PM25_EW}
ewma_state  = {}
for hl in PM25_EW:
    a = ewma_alphas[hl]; v = float(train['pm25_orig'].iloc[0])
    for val in train['pm25_orig'].values[1:]:
        v = a*float(val) + (1-a)*v
    ewma_state[hl] = v

pm25_buf   = deque(train['pm25_orig'].values, maxlen=400)
test_preds = []
t0 = time.time()

for i in range(len(X_test_base)):
    row = X_test_base[i].copy()
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

    r2    = row.reshape(1, -1)
    p_log = wl*lgb_final.predict(r2)[0] + wx*xgb_final.predict(r2)[0] + wh*hgb_final.predict(r2)[0]
    pp    = float(np.expm1(p_log).clip(0))
    test_preds.append(p_log)
    pm25_buf.append(pp)
    for hl in PM25_EW:
        ewma_state[hl] = ewma_alphas[hl]*pp + (1-ewma_alphas[hl])*ewma_state[hl]
    if (i+1) % 1000 == 0:
        log(f"  {i+1}/{len(X_test_base)} rows  last={pp:.1f}")

log(f"  Recursive done in {time.time()-t0:.1f}s")

# ── Save submission ───────────────────────────────────────────────────────────
sub = test[['record_id']].copy()
sub['predicted_pm25'] = np.expm1(np.array(test_preds)).clip(0)
sub.to_csv(SUB, index=False)
shutil.copy(SUB, DL_SUB)
preds = sub['predicted_pm25'].values
log(f"\n{'='*55}")
log(f"V5 COMPLETE")
log(f"  LGB OOF:   {r_lgb_oof:.4f}")
log(f"  XGB OOF:   {r_xgb_oof:.4f}")
log(f"  HGB OOF:   {r_hgb_oof:.4f}")
log(f"  Blend OOF: {r_blend:.4f}  (wL={wl:.3f} wX={wx:.3f} wH={wh:.3f})")
log(f"  Features:  {len(FCOLS)}")
log(f"  Pred stats: min={preds.min():.1f}  mean={preds.mean():.1f}  max={preds.max():.1f}")
log(f"  Saved: {DL_SUB}")
log(f"{'='*55}")

json.dump(dict(lgb=r_lgb_oof, xgb=r_xgb_oof, hgb=r_hgb_oof, blend=r_blend,
               wl=wl, wx=wx, wh=wh, n_features=len(FCOLS)), open(META,'w'), indent=2)
