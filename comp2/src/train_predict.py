"""
NBA Home Team Point Differential Prediction
Competition 2 — RMSE minimization
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "../data"
SUBMIT_DIR = "../submissions"

TRAIN_PATH  = f"{DATA_DIR}/train.csv"
TEST_PATH   = f"{DATA_DIR}/test.csv"
SUBMIT_PATH = f"{SUBMIT_DIR}/submission.csv"

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Train shape: {train.shape}  |  seasons {train.season.min()}-{train.season.max()}")
print(f"Test  shape: {test.shape}   |  seasons {test.season.min()}-{test.season.max()}")

# ── Feature engineering ──────────────────────────────────────────────────────
for df in [train, test]:
    df["elo_diff_sq"]        = df["elo_diff"] ** 2
    df["elo_diff_cb"]        = df["elo_diff"] ** 3
    df["margin_roll5_diff"]  = df["home_margin_roll5"]  - df["away_margin_roll5"]
    df["margin_roll20_diff"] = df["home_margin_roll20"] - df["away_margin_roll20"]
    df["home_net_roll5"]     = df["home_pts_roll5"]  - df["home_opp_pts_roll5"]
    df["away_net_roll5"]     = df["away_pts_roll5"]  - df["away_opp_pts_roll5"]
    df["net_roll5_diff"]     = df["home_net_roll5"]  - df["away_net_roll5"]
    df["home_net_roll10"]    = df["home_pts_roll10"] - df["home_opp_pts_roll10"]
    df["away_net_roll10"]    = df["away_pts_roll10"] - df["away_opp_pts_roll10"]
    df["net_roll10_diff"]    = df["home_net_roll10"] - df["away_net_roll10"]
    df["home_net_roll20"]    = df["home_pts_roll20"] - df["home_opp_pts_roll20"]
    df["away_net_roll20"]    = df["away_pts_roll20"] - df["away_opp_pts_roll20"]
    df["net_roll20_diff"]    = df["home_net_roll20"] - df["away_net_roll20"]
    df["win_roll5_diff"]     = df["home_win_roll5"]  - df["away_win_roll5"]
    df["win_roll10_diff"]    = df["home_win_roll10"] - df["away_win_roll10"]
    df["elo_delta_diff"]     = df["home_elo_delta_prev1"] - df["away_elo_delta_prev1"]
    df["elo_delta_roll5_diff"] = df["home_elo_delta_roll5"] - df["away_elo_delta_roll5"]
    df["any_roll_missing"]   = (
        df["home_roll_missing_10"].astype(int) | df["away_roll_missing_10"].astype(int) |
        df["home_roll_missing_20"].astype(int) | df["away_roll_missing_20"].astype(int)
    )

# ── Features / target ────────────────────────────────────────────────────────
DROP_COLS    = ["id", "home_margin", "season"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]
TARGET       = "home_margin"

X_all  = train[FEATURE_COLS]
y_all  = train[TARGET]
X_test = test[FEATURE_COLS]

print(f"\nFeature count: {len(FEATURE_COLS)}")

# ── Time-based validation split ──────────────────────────────────────────────
VAL_CUTOFF = 2012
mask_tr  = train["season"] < VAL_CUTOFF
mask_val = train["season"] >= VAL_CUTOFF

X_tr,  y_tr  = X_all[mask_tr],  y_all[mask_tr]
X_val, y_val = X_all[mask_val], y_all[mask_val]

print(f"Train rows: {len(X_tr)}  |  Val rows: {len(X_val)}")

# ── LightGBM base params ─────────────────────────────────────────────────────
BASE_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "learning_rate":     0.02,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 30,
    "feature_fraction":  0.80,
    "bagging_fraction":  0.80,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "n_jobs":            -1,
    "verbose":           -1,
}

lgb_train_ds = lgb.Dataset(X_tr,  label=y_tr)
lgb_val_ds   = lgb.Dataset(X_val, label=y_val, reference=lgb_train_ds)
callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]

val_preds  = []
test_preds = []
models     = []

seeds = [42, 123, 7]
for seed in seeds:
    params = {**BASE_PARAMS, "seed": seed}
    print(f"\n-- Training LightGBM seed={seed} --")
    m = lgb.train(params, lgb_train_ds, num_boost_round=5000,
                  valid_sets=[lgb_val_ds], callbacks=callbacks)
    vp = m.predict(X_val)
    tp = m.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_val, vp))
    print(f"   Val RMSE: {rmse:.4f}  |  best iter: {m.best_iteration}")
    val_preds.append(vp);  test_preds.append(tp);  models.append(m)

# Wider tree variant - fresh Dataset to avoid feature_pre_filter conflict
WIDE_PARAMS = {**BASE_PARAMS, "seed": 99, "num_leaves": 127,
               "learning_rate": 0.015, "min_child_samples": 20,
               "feature_pre_filter": False}
lgb_train_w = lgb.Dataset(X_tr,  label=y_tr)
lgb_val_w   = lgb.Dataset(X_val, label=y_val, reference=lgb_train_w)
print("\n-- Training LightGBM seed=99 num_leaves=127 --")
m = lgb.train(WIDE_PARAMS, lgb_train_w, num_boost_round=6000,
              valid_sets=[lgb_val_w], callbacks=callbacks)
vp = m.predict(X_val);  tp = m.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_val, vp))
print(f"   Val RMSE: {rmse:.4f}  |  best iter: {m.best_iteration}")
val_preds.append(vp);  test_preds.append(tp);  models.append(m)

# ── Ensemble ──────────────────────────────────────────────────────────────────
pred_val_ens  = np.mean(val_preds,  axis=0)
pred_test_ens = np.mean(test_preds, axis=0)
rmse_ens = np.sqrt(mean_squared_error(y_val, pred_val_ens))
print(f"\n====  Ensemble Val RMSE ({len(models)}-model avg): {rmse_ens:.4f}  ====")

# ── Feature importance ────────────────────────────────────────────────────────
fi = pd.DataFrame({
    "feature":    models[0].feature_name(),
    "importance": models[0].feature_importance("gain"),
}).sort_values("importance", ascending=False)
print("\nTop-15 features (model 0 gain):")
print(fi.head(15).to_string(index=False))

# ── Save submission ───────────────────────────────────────────────────────────
os.makedirs(SUBMIT_DIR, exist_ok=True)
submission = pd.DataFrame({"id": test["id"], "home_margin": pred_test_ens})
submission.to_csv(SUBMIT_PATH, index=False)

print(f"\nsubmission.csv saved -> {SUBMIT_PATH}")
print(f"  Rows: {len(submission)}  id range: {submission.id.min()}-{submission.id.max()}")
print(submission.head())
