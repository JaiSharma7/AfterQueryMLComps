"""
Online Shoppers Purchase Intention Prediction
==============================================
Bespoke solution with:
  - Two-stage page_values regime modeling
  - Zero-duration anomaly flags
  - Fold-aware target encoding (no leakage)
  - Duration-per-page engagement intensity features
  - LightGBM GBDT + LightGBM DART + XGBoost + HistGradientBoosting weighted ensemble
  - Stratified 5-Fold CV
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
COMP_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(COMP_DIR, "train.csv")
TEST_PATH  = os.path.join(COMP_DIR, "test.csv")
SUB_PATH   = os.path.join(COMP_DIR, "submission.csv")

# ─── Load Data ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data ...")
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)
print(f"  Train shape: {train_raw.shape}  |  Test shape: {test_raw.shape}")

TARGET = "purchased"
ID_COL = "session_id"

# ─── Remove exact duplicates (exclude session_id & target) ────────────────────
dup_cols = [c for c in train_raw.columns if c not in [ID_COL, TARGET]]
n_before = len(train_raw)
train_raw = train_raw.drop_duplicates(subset=dup_cols, keep="first")
print(f"  Dropped {n_before - len(train_raw)} exact duplicates → {len(train_raw)} rows remain")

# ─── Month Normalisation Map ──────────────────────────────────────────────────
MONTH_MAP = {
    "jan": 1,  "january": 1,
    "feb": 2,  "february": 2,
    "mar": 3,  "march": 3,
    "apr": 4,  "april": 4,
    "may": 5,
    "jun": 6,  "june": 6,
    "jul": 7,  "july": 7,
    "aug": 8,  "august": 8,
    "sep": 9,  "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

VISITOR_MAP = {"Returning_Visitor": 0, "New_Visitor": 1, "Other": 2}

# ─── Utilities ────────────────────────────────────────────────────────────────
def safe_div(a, b, fill=0.0):
    denom = np.where(b == 0, np.nan, b)
    result = a / denom
    return np.where(np.isnan(result), fill, result)


def build_te_maps(df_with_target, te_cols, smoothing=5, global_mean=0.155):
    maps = {}
    for col in te_cols:
        stats = df_with_target.groupby(col)[TARGET].agg(["count", "mean"])
        stats.columns = ["n", "mu"]
        stats["te"] = (stats["n"] * stats["mu"] + smoothing * global_mean) / (stats["n"] + smoothing)
        maps[col] = stats["te"].to_dict()
    return maps


def apply_te_maps(df, maps, te_cols, global_mean=0.155):
    df = df.copy()
    for col in te_cols:
        df[f"{col}_te"] = df[col].map(maps.get(col, {})).fillna(global_mean)
    return df


def eng_features(df):
    """Core feature engineering (TE applied separately, fold-aware)."""
    d = df.copy()

    # Month
    d["month_str"] = d["month"].astype(str).str.strip().str.lower()
    d["month_num"] = d["month_str"].map(MONTH_MAP).fillna(0).astype(int)
    d["is_holiday_season"]    = d["month_num"].isin([11, 12]).astype(int)
    d["is_data_scarce_month"] = d["month_num"].isin([0, 1, 4]).astype(int)
    d["month_sin"] = np.sin(2 * np.pi * d["month_num"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month_num"] / 12)

    # page_values
    d["pv_nonzero"] = (d["page_values"] > 0).astype(int)
    d["pv_log1p"]   = np.log1p(d["page_values"])
    d["pv_sqrt"]    = np.sqrt(d["page_values"])
    d["pv_bucket"]  = pd.cut(
        d["page_values"], bins=[-0.001, 0, 5, 25, np.inf], labels=[0, 1, 2, 3]
    ).astype(int)

    # Duration-per-page ratios
    d["admin_dur_per_page"]   = safe_div(d["administrative_duration"],   d["administrative"],   0.0)
    d["info_dur_per_page"]    = safe_div(d["informational_duration"],     d["informational"],    0.0)
    d["product_dur_per_page"] = safe_div(d["product_related_duration"],   d["product_related"],  0.0)

    # Anomaly flags: page visited but zero duration
    d["admin_zero_dur"]   = ((d["administrative"]  > 0) & (d["administrative_duration"]  == 0)).astype(int)
    d["info_zero_dur"]    = ((d["informational"]   > 0) & (d["informational_duration"]   == 0)).astype(int)
    d["product_zero_dur"] = ((d["product_related"] > 0) & (d["product_related_duration"] == 0)).astype(int)
    d["any_zero_dur"]     = ((d["admin_zero_dur"] == 1) | (d["info_zero_dur"] == 1) | (d["product_zero_dur"] == 1)).astype(int)

    # Session aggregates
    d["total_pages"]    = d["administrative"] + d["informational"] + d["product_related"]
    d["total_duration"] = d["administrative_duration"] + d["informational_duration"] + d["product_related_duration"]
    d["avg_page_dur"]   = safe_div(d["total_duration"], d["total_pages"], 0.0)

    # Disengagement signals
    d["bounce_x_exit"]    = d["bounce_rates"] * d["exit_rates"]
    d["exit_minus_bounce"] = d["exit_rates"] - d["bounce_rates"]

    # PV × engagement interactions
    d["pv_x_products"]  = d["pv_log1p"] * np.log1p(d["product_related"])
    d["pv_x_stay"]      = d["pv_log1p"] * (1.0 - d["exit_rates"])

    # Visitor type
    d["visitor_enc"]  = d["visitor_type"].map(VISITOR_MAP).fillna(2).astype(int)
    d["is_returning"] = (d["visitor_enc"] == 0).astype(int)
    d["is_new"]       = (d["visitor_enc"] == 1).astype(int)

    # Composite
    d["weekend_holiday"]   = d["weekend"] * d["is_holiday_season"]
    d["is_special_day"]    = (d["special_day"] > 0).astype(int)
    d["product_pg_ratio"]  = safe_div(d["product_related"], d["total_pages"], 0.0)
    d["admin_pg_ratio"]    = safe_div(d["administrative"],  d["total_pages"], 0.0)

    # Log-transform raw duration columns (heavy right skew)
    for col in ["administrative_duration", "informational_duration", "product_related_duration"]:
        d[f"{col}_log"] = np.log1p(d[col])

    return d


# ─── Feature Sets ─────────────────────────────────────────────────────────────
TE_COLS = ["traffic_type", "region", "browser", "operating_systems"]

RAW_KEEP = [
    "administrative", "administrative_duration",
    "informational", "informational_duration",
    "product_related", "product_related_duration",
    "bounce_rates", "exit_rates", "page_values", "special_day",
    "operating_systems", "browser", "region", "traffic_type",
    "weekend",
]

ENG_COLS = [
    "month_num", "is_holiday_season", "is_data_scarce_month",
    "month_sin", "month_cos",
    "pv_nonzero", "pv_log1p", "pv_sqrt", "pv_bucket",
    "admin_dur_per_page", "info_dur_per_page", "product_dur_per_page",
    "admin_zero_dur", "info_zero_dur", "product_zero_dur", "any_zero_dur",
    "total_pages", "total_duration", "avg_page_dur",
    "bounce_x_exit", "exit_minus_bounce",
    "pv_x_products", "pv_x_stay",
    "visitor_enc", "is_returning", "is_new",
    "weekend_holiday", "is_special_day",
    "product_pg_ratio", "admin_pg_ratio",
    "administrative_duration_log", "informational_duration_log", "product_related_duration_log",
    "traffic_type_te", "region_te", "browser_te", "operating_systems_te",
]

ALL_FEATURES = RAW_KEEP + ENG_COLS

CAT_INDICES_LGB = [ALL_FEATURES.index(c) for c in
                   ["operating_systems", "browser", "region", "traffic_type",
                    "month_num", "pv_bucket", "visitor_enc"] if c in ALL_FEATURES]

# ─── Model Factories ──────────────────────────────────────────────────────────
def make_lgb_gbdt():
    return lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.025, num_leaves=63,
        min_child_samples=20, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=5.45, random_state=42, n_jobs=-1,
        boosting_type="gbdt", verbose=-1,
    )

def make_lgb_dart():
    return lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.04, num_leaves=63,
        min_child_samples=20, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=5.45, random_state=42, n_jobs=-1,
        boosting_type="dart", verbose=-1,
    )

def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=6,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=2.0,
        scale_pos_weight=5.45, random_state=42,
        eval_metric="auc", verbosity=0, n_jobs=-1,
        early_stopping_rounds=60,
    )

def make_hgb():
    """sklearn HistGradientBoosting — handles categoricals natively, fast, robust."""
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
        scoring="roc_auc",
        validation_fraction=0.1,
        n_iter_no_change=30,
    )

# ─── CV Loop ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5-Fold Stratified CV ...")

y_full  = train_raw[TARGET].values
X_raw   = train_raw.drop(columns=[TARGET])
test_df = test_raw.copy()
test_ids = test_df[ID_COL].values

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb_g  = np.zeros(len(y_full))
oof_lgb_d  = np.zeros(len(y_full))
oof_xgb    = np.zeros(len(y_full))
oof_hgb    = np.zeros(len(y_full))

tst_lgb_g  = np.zeros(len(test_df))
tst_lgb_d  = np.zeros(len(test_df))
tst_xgb    = np.zeros(len(test_df))
tst_hgb    = np.zeros(len(test_df))

fold_aucs  = {"lgb_g": [], "lgb_d": [], "xgb": [], "hgb": []}

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_raw, y_full)):
    print(f"\n  ── Fold {fold + 1}/{N_FOLDS} ──────────────────────────────")

    tr_raw = X_raw.iloc[tr_idx].copy()
    vl_raw = X_raw.iloc[vl_idx].copy()
    y_tr   = y_full[tr_idx]
    y_vl   = y_full[vl_idx]

    # Fold-aware target encoding (fit on train split only)
    tr_raw_te = tr_raw.copy()
    tr_raw_te[TARGET] = y_tr
    te_maps = build_te_maps(tr_raw_te, TE_COLS)

    tr_fe = eng_features(apply_te_maps(tr_raw, te_maps, TE_COLS))
    vl_fe = eng_features(apply_te_maps(vl_raw, te_maps, TE_COLS))
    te_fe = eng_features(apply_te_maps(test_df.copy(), te_maps, TE_COLS))

    X_tr  = tr_fe[ALL_FEATURES].values
    X_vl  = vl_fe[ALL_FEATURES].values
    X_te  = te_fe[ALL_FEATURES].values

    # ── LGB GBDT ──────────────────────────────────────────────────────────
    m = make_lgb_gbdt()
    m.fit(X_tr, y_tr,
          eval_set=[(X_vl, y_vl)],
          categorical_feature=CAT_INDICES_LGB,
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)])
    oof_lgb_g[vl_idx] = m.predict_proba(X_vl)[:, 1]
    tst_lgb_g += m.predict_proba(X_te)[:, 1] / N_FOLDS
    fold_aucs["lgb_g"].append(roc_auc_score(y_vl, oof_lgb_g[vl_idx]))
    print(f"    LGB  GBDT   AUC = {fold_aucs['lgb_g'][-1]:.5f}")

    # ── LGB DART ──────────────────────────────────────────────────────────
    m = make_lgb_dart()
    m.fit(X_tr, y_tr, categorical_feature=CAT_INDICES_LGB)
    oof_lgb_d[vl_idx] = m.predict_proba(X_vl)[:, 1]
    tst_lgb_d += m.predict_proba(X_te)[:, 1] / N_FOLDS
    fold_aucs["lgb_d"].append(roc_auc_score(y_vl, oof_lgb_d[vl_idx]))
    print(f"    LGB  DART   AUC = {fold_aucs['lgb_d'][-1]:.5f}")

    # ── XGBoost ───────────────────────────────────────────────────────────
    m = make_xgb()
    m.fit(X_tr, y_tr,
          eval_set=[(X_vl, y_vl)],
          verbose=False)
    oof_xgb[vl_idx] = m.predict_proba(X_vl)[:, 1]
    tst_xgb += m.predict_proba(X_te)[:, 1] / N_FOLDS
    fold_aucs["xgb"].append(roc_auc_score(y_vl, oof_xgb[vl_idx]))
    print(f"    XGBoost     AUC = {fold_aucs['xgb'][-1]:.5f}")

    # ── HistGradientBoosting ───────────────────────────────────────────────
    m = make_hgb()
    m.fit(X_tr, y_tr)
    oof_hgb[vl_idx] = m.predict_proba(X_vl)[:, 1]
    tst_hgb += m.predict_proba(X_te)[:, 1] / N_FOLDS
    fold_aucs["hgb"].append(roc_auc_score(y_vl, oof_hgb[vl_idx]))
    print(f"    HGB         AUC = {fold_aucs['hgb'][-1]:.5f}")

# ─── OOF AUC Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OOF AUC Results:")
model_names  = ["LGB_GBDT", "LGB_DART", "XGBoost", "HGB"]
oof_preds    = [oof_lgb_g, oof_lgb_d, oof_xgb, oof_hgb]
model_aucs   = {}
fold_keys    = ["lgb_g", "lgb_d", "xgb", "hgb"]

for mname, preds, fkey in zip(model_names, oof_preds, fold_keys):
    auc = roc_auc_score(y_full, preds)
    model_aucs[mname] = auc
    fm = np.mean(fold_aucs[fkey])
    fs = np.std(fold_aucs[fkey])
    print(f"  {mname:12s}: OOF AUC = {auc:.5f}  (folds: {fm:.5f} ± {fs:.5f})")

# ─── Softmax-Sharpened Weighted Ensemble ──────────────────────────────────────
auc_arr    = np.array(list(model_aucs.values()))
sharp_exp  = np.exp(auc_arr * 25)
weights    = sharp_exp / sharp_exp.sum()

print("\nEnsemble Weights:")
for mname, w in zip(model_names, weights):
    print(f"  {mname:12s}: {w:.4f}")

oof_ens = sum(w * p for w, p in zip(weights, oof_preds))
tst_ens = sum(w * p for w, p in zip(weights,
                [tst_lgb_g, tst_lgb_d, tst_xgb, tst_hgb]))

ens_auc = roc_auc_score(y_full, oof_ens)
print(f"\n  ENSEMBLE      OOF AUC = {ens_auc:.5f}  ← final training score")

# ─── Submission ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Generating submission ...")
test_probs = np.clip(tst_ens, 1e-6, 1 - 1e-6)

submission = pd.DataFrame({
    "session_id": test_ids,
    "purchase_probability": test_probs,
})

# Sanity checks
assert not submission["purchase_probability"].isna().any(), "NaN detected!"
assert len(submission) == len(test_df), "Row count mismatch!"
assert submission["purchase_probability"].between(0, 1).all(), "Probs out of range!"

submission.to_csv(SUB_PATH, index=False)
print(f"  Saved: {SUB_PATH}")
print(f"  Shape: {submission.shape}")
print(f"  Predicted mean purchase prob: {test_probs.mean():.4f}")
print(f"  Sample:\n{submission.head()}")
print(f"\n✅ Done!  OOF Ensemble AUC = {ens_auc:.5f}")
