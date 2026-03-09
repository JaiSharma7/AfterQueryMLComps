"""
NBA Home Margin Prediction - v4
Three-source prediction blend to close val-to-LB gap:
  A) Val-phase model directly on test (no retraining, no era noise)
  B) Modern-era retrain on 2000-2013 (modern NBA only)
  C) Full-data retrain on 1947-2013 (current approach)
Final test pred = mean(A, B, C) per model, then ensemble across models.
Drop nl=255 (consistently 12.00 on val).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, os
from sklearn.metrics import mean_squared_error

DATA_DIR, SUBMIT_DIR = "../data", "../submissions"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

for df in [train, test]:
    df["elo_diff_sq"]          = df["elo_diff"] ** 2
    df["elo_diff_cb"]          = df["elo_diff"] ** 3
    df["margin_roll5_diff"]    = df["home_margin_roll5"]  - df["away_margin_roll5"]
    df["margin_roll20_diff"]   = df["home_margin_roll20"] - df["away_margin_roll20"]
    df["home_net_roll5"]       = df["home_pts_roll5"]  - df["home_opp_pts_roll5"]
    df["away_net_roll5"]       = df["away_pts_roll5"]  - df["away_opp_pts_roll5"]
    df["net_roll5_diff"]       = df["home_net_roll5"]  - df["away_net_roll5"]
    df["home_net_roll10"]      = df["home_pts_roll10"] - df["home_opp_pts_roll10"]
    df["away_net_roll10"]      = df["away_pts_roll10"] - df["away_opp_pts_roll10"]
    df["net_roll10_diff"]      = df["home_net_roll10"] - df["away_net_roll10"]
    df["home_net_roll20"]      = df["home_pts_roll20"] - df["home_opp_pts_roll20"]
    df["away_net_roll20"]      = df["away_pts_roll20"] - df["away_opp_pts_roll20"]
    df["net_roll20_diff"]      = df["home_net_roll20"] - df["away_net_roll20"]
    df["win_roll5_diff"]       = df["home_win_roll5"]  - df["away_win_roll5"]
    df["win_roll10_diff"]      = df["home_win_roll10"] - df["away_win_roll10"]
    df["elo_delta_diff"]       = df["home_elo_delta_prev1"] - df["away_elo_delta_prev1"]
    df["elo_delta_roll5_diff"] = df["home_elo_delta_roll5"] - df["away_elo_delta_roll5"]
    df["any_roll_missing"]     = (
        df["home_roll_missing_10"].astype(int) | df["away_roll_missing_10"].astype(int) |
        df["home_roll_missing_20"].astype(int) | df["away_roll_missing_20"].astype(int))
    df["is_modern"] = (df["season"] >= 1980).astype(int)

FEATURE_COLS = [c for c in train.columns if c not in ["id","home_margin","season"]]
X_all  = train[FEATURE_COLS];  y_all = train["home_margin"]
X_test = test[FEATURE_COLS]

# Three data pools for test predictions
modern_mask  = train["season"] >= 2000           # B: 2000-2013 (~18k games)
X_mod, y_mod = X_all[modern_mask], y_all[modern_mask]

# Val split
mask_tr  = train["season"] < 2012
mask_val = train["season"] >= 2012
X_tr, y_tr   = X_all[mask_tr],  y_all[mask_tr]
X_val, y_val = X_all[mask_val], y_all[mask_val]
print(f"Train {X_tr.shape} | Val {X_val.shape} | Modern {X_mod.shape} | Full {X_all.shape}")

val_preds, test_preds, val_rmses = [], [], []

LGB_CONFIGS = [
    dict(seed=42,  num_leaves=63,  lr=0.02,  rounds=5000, mcs=30),
    dict(seed=123, num_leaves=63,  lr=0.02,  rounds=5000, mcs=30),
    dict(seed=7,   num_leaves=63,  lr=0.02,  rounds=5000, mcs=30),
    dict(seed=17,  num_leaves=63,  lr=0.02,  rounds=5000, mcs=30),
    dict(seed=31,  num_leaves=95,  lr=0.018, rounds=5000, mcs=25),
    dict(seed=99,  num_leaves=127, lr=0.015, rounds=6000, mcs=20),
    dict(seed=55,  num_leaves=127, lr=0.015, rounds=6000, mcs=20),
    dict(seed=201, num_leaves=63,  lr=0.01,  rounds=8000, mcs=30),
    dict(seed=202, num_leaves=63,  lr=0.01,  rounds=8000, mcs=30),
]

LGB_BASE = dict(objective="regression", metric="rmse",
                feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=5,
                reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1,
                feature_pre_filter=False)
cb = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(9999)]

print("\n=== LightGBM ===")
for cfg in LGB_CONFIGS:
    p = {**LGB_BASE, "seed": cfg["seed"], "num_leaves": cfg["num_leaves"],
         "learning_rate": cfg["lr"], "min_child_samples": cfg["mcs"]}

    # A: val-phase model -> directly predict test (no era noise)
    ds_tr  = lgb.Dataset(X_tr,  label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    m_a = lgb.train(p, ds_tr, num_boost_round=cfg["rounds"],
                    valid_sets=[ds_val], callbacks=cb)
    best = m_a.best_iteration
    vp   = m_a.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, vp))
    tp_a = m_a.predict(X_test)

    # B: modern-era retrain (2000-2013)
    full_iter = max(best, int(best * 1.05))
    m_b = lgb.train(p, lgb.Dataset(X_mod, label=y_mod), num_boost_round=full_iter)
    tp_b = m_b.predict(X_test)

    # C: full-data retrain (1947-2013)
    m_c = lgb.train(p, lgb.Dataset(X_all, label=y_all), num_boost_round=full_iter)
    tp_c = m_c.predict(X_test)

    # Blend: A + B + C equally
    tp_blend = (tp_a + tp_b + tp_c) / 3.0
    print(f"  seed={cfg['seed']:3d} nl={cfg['num_leaves']:3d} val={rmse:.4f} iter={best}")
    val_preds.append(vp); val_rmses.append(rmse); test_preds.append(tp_blend)

XGB_CONFIGS = [
    dict(seed=42,  max_depth=6, lr=0.02,  rounds=5000, mcw=30, ss=0.80, cs=0.80),
    dict(seed=123, max_depth=5, lr=0.02,  rounds=5000, mcw=25, ss=0.80, cs=0.80),
    dict(seed=7,   max_depth=7, lr=0.015, rounds=6000, mcw=20, ss=0.80, cs=0.70),
    dict(seed=99,  max_depth=6, lr=0.01,  rounds=8000, mcw=30, ss=0.80, cs=0.80),
]
print("\n=== XGBoost ===")
for cfg in XGB_CONFIGS:
    p = dict(objective="reg:squarederror", eval_metric="rmse",
             learning_rate=cfg["lr"], max_depth=cfg["max_depth"],
             min_child_weight=cfg["mcw"], subsample=cfg["ss"],
             colsample_bytree=cfg["cs"], reg_alpha=0.1, reg_lambda=1.0,
             seed=cfg["seed"], nthread=-1, verbosity=0)
    dtrain = xgb.DMatrix(X_tr,  label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    dall   = xgb.DMatrix(X_all, label=y_all)
    dmod   = xgb.DMatrix(X_mod, label=y_mod)
    dtest  = xgb.DMatrix(X_test)

    m_a = xgb.train(p, dtrain, num_boost_round=cfg["rounds"],
                    evals=[(dval, "val")], early_stopping_rounds=100,
                    verbose_eval=False)
    best = m_a.best_iteration + 1
    vp   = m_a.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, vp))
    tp_a = m_a.predict(dtest)

    full_iter = max(best, int(best * 1.05))
    tp_b = xgb.train(p, dmod, num_boost_round=full_iter, verbose_eval=False).predict(dtest)
    tp_c = xgb.train(p, dall, num_boost_round=full_iter, verbose_eval=False).predict(dtest)
    tp_blend = (tp_a + tp_b + tp_c) / 3.0

    print(f"  seed={cfg['seed']:3d} md={cfg['max_depth']} val={rmse:.4f} iter={best}")
    val_preds.append(vp); val_rmses.append(rmse); test_preds.append(tp_blend)

# Weighted ensemble
val_rmses = np.array(val_rmses)
weights = 1.0 / (val_rmses ** 2)
weights /= weights.sum()
pred_val_ens  = np.average(val_preds,  axis=0, weights=weights)
pred_test_ens = np.average(test_preds, axis=0, weights=weights)
rmse_ens = np.sqrt(mean_squared_error(y_val, pred_val_ens))
print(f"\n==== Weighted Ensemble ({len(val_preds)} models): val RMSE {rmse_ens:.4f} ====")
print("RMSEs:", sorted([f"{r:.4f}" for r in val_rmses]))

os.makedirs(SUBMIT_DIR, exist_ok=True)
sub = pd.DataFrame({"id": test["id"], "home_margin": pred_test_ens})
sub.to_csv(f"{SUBMIT_DIR}/submission.csv", index=False)
print(f"Saved {len(sub)} rows | pred mean={pred_test_ens.mean():.3f} std={pred_test_ens.std():.3f}")
