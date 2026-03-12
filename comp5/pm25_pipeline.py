"""Beijing Air Quality PM2.5 Forecasting - Fast Optimized Pipeline"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os, json, time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR  = r"C:\Users\jaish\OneDrive - The University of Texas at Austin\Documents\AfterQueryMLComps\comp5"
LOG_FILE = os.path.join(OUT_DIR, "pipeline_log.txt")
SUBMISSION = os.path.join(OUT_DIR, "submission.csv")

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f: f.write(msg + '\n')

def rmse_orig(y_true, y_pred_log):
    return np.sqrt(mean_squared_error(y_true, np.expm1(y_pred_log).clip(0)))

# ── Load ─────────────────────────────────────────────────────────────────────
log("="*55 + "\nSTEP 1: Loading")
train = pd.read_csv(r"C:\Users\jaish\Downloads\train (5).csv", parse_dates=['datetime'])
test  = pd.read_csv(r"C:\Users\jaish\Downloads\test (5).csv",  parse_dates=['datetime'])
train = train.sort_values('datetime').reset_index(drop=True)
test  = test.sort_values('datetime').reset_index(drop=True)
log(f"  Train {train.shape}, Test {test.shape}")

# ── Target ───────────────────────────────────────────────────────────────────
train['pm25_original'] = train['pm25'].clip(0)
train['pm25_log']      = np.log1p(train['pm25_original'])

# ── Wind ─────────────────────────────────────────────────────────────────────
WIND = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,
        'SSE':157.5,'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,
        'WNW':292.5,'NW':315,'NNW':337.5}
for df in [train, test]:
    df['wind_dir_deg'] = df['wind_direction'].map(WIND)
    df['wind_dir_rad'] = np.radians(df['wind_dir_deg'])
    df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
    df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])

# ── Temporal ──────────────────────────────────────────────────────────────────
for df in [train, test]:
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    dow = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2*np.pi*dow/7)
    df['dow_cos'] = np.cos(2*np.pi*dow/7)
    doy = df['datetime'].dt.dayofyear
    df['doy_sin'] = np.sin(2*np.pi*doy/365)
    df['doy_cos'] = np.cos(2*np.pi*doy/365)
    df['is_weekend']   = (dow >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
log("Steps 2-4 done")

# ── Combined lag/rolling (NO pm25) ────────────────────────────────────────────
BASE = ['pm10','so2','no2','co','o3','temperature','pressure','dew_point','wind_speed','rain']
t_c = train.drop(columns=['pm25','pm25_original','pm25_log'])
t_c['is_test'] = False
te_c = test.copy(); te_c['is_test'] = True
comb = pd.concat([t_c, te_c], ignore_index=True).sort_values('datetime').reset_index(drop=True)
assert 'pm25' not in comb.columns

log("Step 5: Imputing + lags + rolling...")
for col in BASE + ['wind_dir_deg','wind_dir_rad','wind_dir_sin','wind_dir_cos']:
    comb[col] = comb[col].interpolate(method='linear', limit=6, limit_direction='both').ffill().bfill()

for col in BASE:
    for lag in [1,2,3,6,12,24,48,72,168]:
        comb[f'{col}_lag_{lag}h'] = comb[col].shift(lag)

for col in BASE:
    for w in [3,6,12,24,48,168]:
        r = comb[col].rolling(w, min_periods=1)
        comb[f'{col}_roll_{w}h_mean'] = r.mean()
        comb[f'{col}_roll_{w}h_std']  = r.std()
        comb[f'{col}_roll_{w}h_min']  = r.min()
        comb[f'{col}_roll_{w}h_max']  = r.max()

comb['humidity_proxy']     = comb['dew_point'] - comb['temperature']
comb['wind_u']             = comb['wind_speed'] * np.sin(comb['wind_dir_rad'])
comb['wind_v']             = comb['wind_speed'] * np.cos(comb['wind_dir_rad'])
comb['pressure_delta_24h'] = comb['pressure'] - comb['pressure_lag_24h']

trn = comb[~comb['is_test']].drop(columns=['is_test']).reset_index(drop=True)
tst = comb[ comb['is_test']].drop(columns=['is_test']).reset_index(drop=True)
trn['pm25_original'] = train['pm25_original'].values
trn['pm25_log']      = train['pm25_log'].values
log(f"  train={trn.shape}, test={tst.shape}")

# ── Feature cols ──────────────────────────────────────────────────────────────
EXCL = {'record_id','datetime','pm25','pm25_original','pm25_log',
        'is_test','wind_direction','wind_dir_deg','wind_dir_rad'}
FCOLS = [c for c in trn.columns if c not in EXCL]
log(f"  {len(FCOLS)} features")

X  = trn[FCOLS].values
y  = trn['pm25_log'].values
yo = trn['pm25_original'].values
Xt = tst[FCOLS].values

tscv = TimeSeriesSplit(n_splits=5)

def cv_oof(model_fn, name):
    """Run 5-fold CV, return (oof_array_on_val_indices, per_fold_rmse, val_mask)"""
    oof  = np.full(len(X), np.nan)
    rmses = []
    best_iters = []
    for fold, (tr_i, val_i) in enumerate(tscv.split(X)):
        m = model_fn(fold)
        m.fit(X[tr_i], y[tr_i], **fit_kwargs(m, X[val_i], y[val_i]))
        pred = m.predict(X[val_i])
        oof[val_i] = pred
        r = rmse_orig(yo[val_i], pred)
        rmses.append(r)
        bi = getattr(m, 'best_iteration_', None) or getattr(m, 'best_iteration', None) or 0
        best_iters.append(bi)
        log(f"    {name} Fold {fold+1}: RMSE={r:.4f}  iter={bi}")
    mask = ~np.isnan(oof)
    oof_rmse = rmse_orig(yo[mask], oof[mask])
    valid_iters = [b for b in best_iters if b and b > 0]
    avg_iter = int(np.mean(valid_iters)) if valid_iters else 0
    log(f"  {name} OOF RMSE: {oof_rmse:.4f}  avg_best_iter={avg_iter}")
    return oof, oof_rmse, best_iters

def fit_kwargs(m, Xv, yv):
    """Return fit() kwargs based on model type"""
    if isinstance(m, lgb.LGBMRegressor):
        return dict(eval_set=[(Xv, yv)],
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    if isinstance(m, xgb.XGBRegressor):
        return dict(eval_set=[(Xv, yv)], verbose=False)
    return {}  # HGBR needs no eval_set

# ── LightGBM ──────────────────────────────────────────────────────────────────
log("\nSTEP 9: LightGBM CV")
t0 = time.time()
lgbm_p = dict(n_estimators=3000, learning_rate=0.05, num_leaves=127,
              min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, reg_alpha=0.1, reg_lambda=1.0,
              n_jobs=-1, random_state=42, verbose=-1)
oof_lgbm, lgbm_rmse, lgbm_iters = cv_oof(lambda f: lgb.LGBMRegressor(**lgbm_p), "LGB")
log(f"  LGB done in {time.time()-t0:.0f}s")

# ── XGBoost ───────────────────────────────────────────────────────────────────
log("\nSTEP 10: XGBoost CV")
t0 = time.time()
lgbm_mean_iter = max(100, int(np.nanmean([b for b in lgbm_iters if b > 0])))
xgb_p = dict(n_estimators=lgbm_mean_iter + 200, learning_rate=0.05, max_depth=6,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
             reg_alpha=0.1, reg_lambda=1.0, tree_method='hist',
             early_stopping_rounds=100,
             n_jobs=-1, random_state=42, verbosity=0)
oof_xgb, xgb_rmse, xgb_iters = cv_oof(lambda f: xgb.XGBRegressor(**xgb_p), "XGB")
log(f"  XGB done in {time.time()-t0:.0f}s")

# ── HGBR ─────────────────────────────────────────────────────────────────────
log("\nSTEP 11: HGBR CV")
t0 = time.time()
hgb_p = dict(max_iter=lgbm_mean_iter, learning_rate=0.05, max_leaf_nodes=127,
             min_samples_leaf=20, l2_regularization=1.0,
             early_stopping=False, random_state=42)
oof_hgb, hgb_rmse, _ = cv_oof(lambda f: HistGradientBoostingRegressor(**hgb_p), "HGB")
log(f"  HGB done in {time.time()-t0:.0f}s")

# ── Optuna LightGBM tuning (25 trials) ───────────────────────────────────────
log("\nSTEP 12: Optuna LightGBM tuning (25 trials)...")
t0 = time.time()

def lgbm_obj(trial):
    p = dict(
        n_estimators=lgbm_mean_iter + 200,
        learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int('nl', 31, 255),
        min_child_samples=trial.suggest_int('mcs', 5, 100),
        feature_fraction=trial.suggest_float('ff', 0.5, 1.0),
        bagging_fraction=trial.suggest_float('bf', 0.5, 1.0),
        bagging_freq=1,
        reg_alpha=trial.suggest_float('ra', 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float('rl', 1e-4, 10.0, log=True),
        n_jobs=-1, random_state=42, verbose=-1
    )
    rs = []
    for tr_i, val_i in tscv.split(X):
        m = lgb.LGBMRegressor(**p)
        m.fit(X[tr_i], y[tr_i],
              eval_set=[(X[val_i], y[val_i])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        rs.append(rmse_orig(yo[val_i], m.predict(X[val_i])))
    return np.mean(rs)

study = optuna.create_study(direction='minimize',
    storage=f'sqlite:///{OUT_DIR}/study.db', study_name='lgbm_pm25',
    load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
study.optimize(lgbm_obj, n_trials=25)
log(f"  Best Optuna RMSE: {study.best_value:.4f}  ({time.time()-t0:.0f}s)")
log(f"  Params: {study.best_params}")

# Re-run OOF with tuned params
best_p = study.best_params.copy()
best_p.update({'n_estimators': lgbm_mean_iter+200, 'bagging_freq':1,
               'n_jobs':-1, 'random_state':42, 'verbose':-1})
oof_lgbm_tuned, lgbm_tuned_rmse, lgbm_tuned_iters = cv_oof(lambda f: lgb.LGBMRegressor(**best_p), "LGB_tuned")
if lgbm_tuned_rmse < lgbm_rmse:
    oof_lgbm = oof_lgbm_tuned; lgbm_rmse = lgbm_tuned_rmse; lgbm_iters = lgbm_tuned_iters
    log("  Using TUNED LightGBM"); lgbm_p = best_p
else:
    log("  Keeping DEFAULT LightGBM (tuning did not improve)")

# ── Ensemble weights ──────────────────────────────────────────────────────────
log("\nSTEP 13: Ensemble...")
mask = ~(np.isnan(oof_lgbm) | np.isnan(oof_xgb) | np.isnan(oof_hgb))
rmse_eq = rmse_orig(yo[mask], ((oof_lgbm+oof_xgb+oof_hgb)/3)[mask])
log(f"  Equal blend RMSE: {rmse_eq:.4f}")

def blend_obj(trial):
    w = np.array([trial.suggest_float(f'w{i}',0,1) for i in range(3)])
    w /= w.sum() + 1e-9
    b = w[0]*oof_lgbm[mask] + w[1]*oof_xgb[mask] + w[2]*oof_hgb[mask]
    return rmse_orig(yo[mask], b)

bs = optuna.create_study(direction='minimize')
bs.optimize(blend_obj, n_trials=300)
bw = np.array([bs.best_params[f'w{i}'] for i in range(3)])
bw /= bw.sum()
rmse_blend = bs.best_value
log(f"  Optuna blend RMSE: {rmse_blend:.4f}  LGB:{bw[0]:.3f} XGB:{bw[1]:.3f} HGB:{bw[2]:.3f}")
w = bw if rmse_blend < rmse_eq - 0.5 else np.array([1/3,1/3,1/3])
log(f"  Final weights: LGB={w[0]:.3f} XGB={w[1]:.3f} HGB={w[2]:.3f}")

# ── Final training + predictions ──────────────────────────────────────────────
log("\nSTEP 14: Final models on full training data...")
final_n = max(200, int(np.nanmean([b for b in lgbm_iters if b>0])))

lgbm_p['n_estimators'] = final_n
mf_lgb = lgb.LGBMRegressor(**lgbm_p); mf_lgb.fit(X, y)
tp_lgb = mf_lgb.predict(Xt)

xgb_p2 = {k:v for k,v in xgb_p.items() if k != 'early_stopping_rounds'}
xgb_p2['n_estimators'] = max(200, int(np.nanmean([b for b in xgb_iters if b>0]) or final_n))
mf_xgb = xgb.XGBRegressor(**xgb_p2); mf_xgb.fit(X, y)
tp_xgb = mf_xgb.predict(Xt)

hgb_p['max_iter'] = final_n
mf_hgb = HistGradientBoostingRegressor(**hgb_p); mf_hgb.fit(X, y)
tp_hgb = mf_hgb.predict(Xt)

final = np.expm1(w[0]*tp_lgb + w[1]*tp_xgb + w[2]*tp_hgb).clip(0)
assert len(final)==6828 and not np.any(np.isnan(final))

sub = pd.DataFrame({'record_id': tst['record_id'].values, 'predicted_pm25': final})
sub.sort_values('record_id').to_csv(SUBMISSION, index=False)
log(f"  Saved: {SUBMISSION}")
log(f"\n{'='*55}")
log("COMPLETE")
log(f"  LGB RMSE:      {lgbm_rmse:.4f}")
log(f"  XGB RMSE:      {xgb_rmse:.4f}")
log(f"  HGB RMSE:      {hgb_rmse:.4f}")
log(f"  Ensemble RMSE: {min(rmse_eq, rmse_blend):.4f}")
log(f"  Pred range: [{final.min():.1f}, {final.max():.1f}]  mean={final.mean():.2f}")
log("="*55)

meta = {'lgbm_rmse':lgbm_rmse,'xgb_rmse':xgb_rmse,'hgb_rmse':hgb_rmse,
        'ensemble_rmse':min(rmse_eq,rmse_blend),'weights':w.tolist(),'n_features':len(FCOLS)}
with open(os.path.join(OUT_DIR,'meta.json'),'w') as f: json.dump(meta, f, indent=2)
