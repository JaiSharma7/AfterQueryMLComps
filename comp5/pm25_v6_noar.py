"""
V6-NoAR: Pure exogenous model (NO PM2.5 autoregressive lags).
No recursive prediction needed at test time → no error accumulation.
This should close the local→LB gap dramatically.
Then we blend V5 (AR) + V6 (no-AR) for the best of both.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os, time, json, shutil
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT    = r"C:\Users\jaish\OneDrive - The University of Texas at Austin\Documents\AfterQueryMLComps\comp5"
DL     = r"C:\Users\jaish\Downloads"
LOG    = os.path.join(OUT, "v6_log.txt")
META   = os.path.join(OUT, "v6_meta.json")
SUB6   = os.path.join(OUT, "submission_v6.csv")
DL_SUB = os.path.join(DL,  "submission_v6.csv")

open(LOG,'w').close()
def log(m): print(m, flush=True); open(LOG,'a').write(m+'\n')
def rmse(yt, yp): return float(np.sqrt(mean_squared_error(yt, np.expm1(yp).clip(0))))

# ── Data ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(DL,"train (5).csv"), parse_dates=['datetime'])
test  = pd.read_csv(os.path.join(DL,"test (5).csv"),  parse_dates=['datetime'])
train = train.sort_values('datetime').reset_index(drop=True)
test  = test.sort_values('datetime').reset_index(drop=True)
train['pm25_orig'] = train['pm25'].clip(lower=0)
train['pm25_log']  = np.log1p(train['pm25_orig'])
log(f"V6-NoAR | Train={len(train)} Test={len(test)}")

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
    df['year_frac']         = df['year'] + (doy-1)/365.0  # continuous time trend
    return df

train = add_temporal(train)
test  = add_temporal(test)

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

# Pressure trends (exogenous — no PM2.5 needed)
for h in [1, 3, 6, 12, 24]:
    comb[f'pressure_d{h}'] = comb['pressure'] - comb['pressure'].shift(h)
for h in [1, 3, 6, 12, 24]:
    comb[f'temp_d{h}'] = comb['temperature'] - comb['temperature'].shift(h)

# ── EXOGENOUS lags + rolling + EWMA (no PM2.5!) ────────────────────────────
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

log(f"Features built. comb shape: {comb.shape}")

# ── Split ─────────────────────────────────────────────────────────────────────
trn_exog = comb[~comb['_t']].drop(columns=['_t']).reset_index(drop=True)
tst_exog = comb[ comb['_t']].drop(columns=['_t']).reset_index(drop=True)
trn_exog['pm25_orig'] = train['pm25_orig'].values
trn_exog['pm25_log']  = train['pm25_log'].values

EXCL = {'record_id','datetime','pm25','pm25_orig','pm25_log','_t',
        'wind_direction','wind_dir_deg','wind_dir_rad'}
FCOLS = [c for c in trn_exog.columns if c not in EXCL]
log(f"Features (no AR): {len(FCOLS)}")

X      = trn_exog[FCOLS].values.astype(np.float32)
y      = trn_exog['pm25_log'].values
yo     = trn_exog['pm25_orig'].values
X_test = tst_exog[FCOLS].values.astype(np.float32)

# ── Load V2 LGB params ─────────────────────────────────────────────────────
try:
    v2s = optuna.load_study(study_name='lgbm_v2', storage=f'sqlite:///{OUT}/study_v2.db')
    bp  = v2s.best_params
    log(f"Loaded V2 params (RMSE={v2s.best_value:.4f})")
except:
    bp = {}

lgb_p = dict(
    learning_rate     = bp.get('lr', 0.04),
    num_leaves        = int(bp.get('nl', 349)),
    min_child_samples = int(bp.get('mcs', 32)),
    feature_fraction  = bp.get('ff', 0.95),
    bagging_fraction  = bp.get('bf', 0.84),
    reg_alpha         = bp.get('ra', 0.01),
    reg_lambda        = bp.get('rl', 0.16),
    min_split_gain    = bp.get('msg', 0.04),
    n_estimators=3000, bagging_freq=1, n_jobs=-1, random_state=42, verbose=-1
)
xgb_p = dict(n_estimators=3000, learning_rate=0.04, max_depth=7,
             min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
             reg_alpha=0.05, reg_lambda=1.0, tree_method='hist',
             n_jobs=-1, random_state=42, verbosity=0, early_stopping_rounds=100)
hgb_p = dict(max_iter=500, learning_rate=0.025, max_leaf_nodes=127,
             min_samples_leaf=25, l2_regularization=0.5, max_depth=8,
             early_stopping=False, random_state=42)

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
oof_l = np.full(len(X), np.nan)
oof_x = np.full(len(X), np.nan)
oof_h = np.full(len(X), np.nan)
lgb_iters, xgb_iters = [], []
t_cv = time.time()
log("\nSTEP 1: 5-fold CV (no AR)")

for fold, (tri, vali) in enumerate(tscv.split(X)):
    ft = time.time()
    m_l = lgb.LGBMRegressor(**lgb_p)
    m_l.fit(X[tri], y[tri], eval_set=[(X[vali], y[vali])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof_l[vali] = m_l.predict(X[vali]); lgb_iters.append(m_l.best_iteration_)
    rl = rmse(yo[vali], oof_l[vali])

    m_x = xgb.XGBRegressor(**xgb_p)
    m_x.fit(X[tri], y[tri], eval_set=[(X[vali], y[vali])], verbose=False)
    oof_x[vali] = m_x.predict(X[vali]); xgb_iters.append(m_x.best_iteration)
    rx = rmse(yo[vali], oof_x[vali])

    m_h = HistGradientBoostingRegressor(**hgb_p)
    m_h.fit(X[tri], y[tri])
    oof_h[vali] = m_h.predict(X[vali])
    rh = rmse(yo[vali], oof_h[vali])

    log(f"  fold {fold+1}: LGB={rl:.4f}(n={m_l.best_iteration_})  "
        f"XGB={rx:.4f}(n={m_x.best_iteration})  HGB={rh:.4f}  ({time.time()-ft:.0f}s)")

mask = ~(np.isnan(oof_l)|np.isnan(oof_x)|np.isnan(oof_h))
rl_oof = rmse(yo[mask], oof_l[mask])
rx_oof = rmse(yo[mask], oof_x[mask])
rh_oof = rmse(yo[mask], oof_h[mask])
log(f"\n  LGB OOF: {rl_oof:.4f}  XGB OOF: {rx_oof:.4f}  HGB OOF: {rh_oof:.4f}")
log(f"  CV total: {(time.time()-t_cv)/60:.1f}min")

# Optuna blend
def obj(trial):
    wl = trial.suggest_float('wl',0.0,1.0)
    wx = trial.suggest_float('wx',0.0,1.0-wl)
    wh = 1.0-wl-wx
    return rmse(yo[mask], wl*oof_l[mask]+wx*oof_x[mask]+wh*oof_h[mask])

bs = optuna.create_study(direction='minimize')
bs.optimize(obj, n_trials=600, n_jobs=1)
wl, wx = bs.best_params['wl'], bs.best_params['wx']
wh = 1.0-wl-wx
r_blend = bs.best_value
log(f"  Blend OOF: {r_blend:.4f}  wL={wl:.3f} wX={wx:.3f} wH={wh:.3f}")

# ── Final models ──────────────────────────────────────────────────────────────
log("\nSTEP 2: Final models on all training data ...")
n_lgb = max(200, int(np.mean([b for b in lgb_iters if b and b>0])))
n_xgb = max(200, int(np.mean([b for b in xgb_iters if b and b>0])))

lgb_p['n_estimators'] = n_lgb
lgb_f = lgb.LGBMRegressor(**lgb_p); lgb_f.fit(X, y)

xp = {k:v for k,v in xgb_p.items() if k!='early_stopping_rounds'}
xp['n_estimators'] = n_xgb
xgb_f = xgb.XGBRegressor(**xp); xgb_f.fit(X, y)

hgb_f = HistGradientBoostingRegressor(**hgb_p); hgb_f.fit(X, y)
log("  Final models done")

# ── Test prediction (no recursion!) ──────────────────────────────────────────
log("\nSTEP 3: Test prediction (direct, no recursion) ...")
plog = wl*lgb_f.predict(X_test) + wx*xgb_f.predict(X_test) + wh*hgb_f.predict(X_test)
preds = np.expm1(plog).clip(0)

sub = test[['record_id']].copy()
sub['predicted_pm25'] = preds
sub.to_csv(SUB6, index=False)
shutil.copy(SUB6, DL_SUB)

log(f"\n{'='*55}")
log(f"V6-NoAR COMPLETE")
log(f"  LGB OOF:   {rl_oof:.4f}")
log(f"  XGB OOF:   {rx_oof:.4f}")
log(f"  HGB OOF:   {rh_oof:.4f}")
log(f"  Blend OOF: {r_blend:.4f}  (wL={wl:.3f} wX={wx:.3f} wH={wh:.3f})")
log(f"  Features:  {len(FCOLS)}")
log(f"  Pred stats: min={preds.min():.1f}  mean={preds.mean():.1f}  max={preds.max():.1f}")
log(f"  Saved: {DL_SUB}")
log(f"{'='*55}")

json.dump(dict(lgb=rl_oof, xgb=rx_oof, hgb=rh_oof, blend=r_blend,
               wl=wl, wx=wx, wh=wh, n_features=len(FCOLS)), open(META,'w'), indent=2)
