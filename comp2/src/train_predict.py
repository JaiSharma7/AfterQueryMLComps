"""
NBA Home Margin Prediction — Optimized v2
Key improvement: retrain on FULL data using best_iter found via time-split val.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb, os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR, SUBMIT_DIR = "../data", "../submissions"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"Train {train.shape} | Test {test.shape}")

# ── Feature engineering ───────────────────────────────────────────────────────
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
        df["home_roll_missing_20"].astype(int) | df["away_roll_missing_20"].astype(int)
    )
    # Recent-era flag (modern 3-point era scoring is very different from 1950s)
    df["is_modern"] = (df["season"] >= 1980).astype(int)

FEATURE_COLS = [c for c in train.columns if c not in ["id","home_margin","season"]]
X_all  = train[FEATURE_COLS];  y_all = train["home_margin"]
X_test = test[FEATURE_COLS]
print(f"Features: {len(FEATURE_COLS)}")

# ── Time-based split ──────────────────────────────────────────────────────────
mask_tr  = train["season"] < 2012
mask_val = train["season"] >= 2012
X_tr, y_tr   = X_all[mask_tr],  y_all[mask_tr]
X_val, y_val = X_all[mask_val], y_all[mask_val]
print(f"Val split -> train {len(X_tr)} | val {len(X_val)}")

# ── LGB configs ───────────────────────────────────────────────────────────────
CONFIGS = [
    dict(seed=42,  num_leaves=63,  lr=0.02,  rounds=5000, mcs=30),
    dict(seed=123, num_leaves=63,  lr=0.02,  rounds=5000, mcs=30),
    dict(seed=7,   num_leaves=63,  lr=0.02,  rounds=5000, mcs=30),
    dict(seed=17,  num_leaves=63,  lr=0.02,  rounds=5000, mcs=30),
    dict(seed=31,  num_leaves=95,  lr=0.018, rounds=5000, mcs=25),
    dict(seed=99,  num_leaves=127, lr=0.015, rounds=6000, mcs=20),
    dict(seed=55,  num_leaves=127, lr=0.015, rounds=6000, mcs=20),
]

BASE = dict(objective="regression", metric="rmse", feature_fraction=0.80,
            bagging_fraction=0.80, bagging_freq=5, reg_alpha=0.1,
            reg_lambda=1.0, n_jobs=-1, verbose=-1)

val_preds, test_preds = [], []
cb = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]

for cfg in CONFIGS:
    p = {**BASE, "seed": cfg["seed"], "num_leaves": cfg["num_leaves"],
         "learning_rate": cfg["lr"], "min_child_samples": cfg["mcs"],
         "feature_pre_filter": False}

    # ① find best iteration on val split
    ds_tr  = lgb.Dataset(X_tr,  label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    m = lgb.train(p, ds_tr, num_boost_round=cfg["rounds"],
                  valid_sets=[ds_val], callbacks=cb)
    best = m.best_iteration
    vp   = m.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, vp))
    print(f"seed={cfg['seed']:3d} nl={cfg['num_leaves']} val={rmse:.4f} iter={best}")
    val_preds.append(vp)

    # ② retrain on ALL train data with best_iter * 1.05 (more data → more rounds)
    full_iter = max(best, int(best * 1.05))
    ds_full = lgb.Dataset(X_all, label=y_all)
    m_full  = lgb.train(p, ds_full, num_boost_round=full_iter)
    test_preds.append(m_full.predict(X_test))

# ── Ridge on elo features (fast, linear signal) ───────────────────────────────
elo_feats = ["elo_diff","elo_diff_sq","elo_diff_cb","elo_delta_diff",
             "elo_delta_roll5_diff","home_elo_pre","away_elo_pre",
             "home_elo_delta_prev1","away_elo_delta_prev1",
             "home_elo_delta_roll5","away_elo_delta_roll5"]
sc = StandardScaler()
Xr_tr   = sc.fit_transform(X_tr[elo_feats])
Xr_val  = sc.transform(X_val[elo_feats])
Xr_all  = sc.transform(X_all[elo_feats])
Xr_test = sc.transform(X_test[elo_feats])

for alpha in [0.1, 1.0, 10.0]:
    r = Ridge(alpha=alpha).fit(Xr_tr, y_tr)
    vp = r.predict(Xr_val)
    rmse = np.sqrt(mean_squared_error(y_val, vp))
    print(f"Ridge a={alpha} val={rmse:.4f}")
    val_preds.append(vp)
    r_full = Ridge(alpha=alpha).fit(Xr_all, y_all)
    test_preds.append(r_full.predict(Xr_test))

# ── Ensemble ──────────────────────────────────────────────────────────────────
pred_val_ens  = np.mean(val_preds,  axis=0)
pred_test_ens = np.mean(test_preds, axis=0)
rmse_ens = np.sqrt(mean_squared_error(y_val, pred_val_ens))
print(f"\n==== Ensemble Val RMSE ({len(val_preds)} models): {rmse_ens:.4f} ====")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(SUBMIT_DIR, exist_ok=True)
sub = pd.DataFrame({"id": test["id"], "home_margin": pred_test_ens})
sub.to_csv(f"{SUBMIT_DIR}/submission.csv", index=False)
print(f"Saved {len(sub)} rows. id range {sub.id.min()}-{sub.id.max()}")
