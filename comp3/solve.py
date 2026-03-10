"""
Comp 3: Air Compressor Failure Prediction - Binary F1 score (positive = 1)

Key insight from EDA:
  - Failure precursor = compressor frozen at idle state
    (Motor_current stuck ~0.04, all sensors constant for hours)
  - vs normal: cycling between idle/active every ~20min
  - Other frozen periods exist (April 20: active-state frozen) - 
    model distinguishes using raw values + rolling variance features
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

TRAIN_PATH = "C:/Users/jaish/Downloads/train (2).csv"
TEST_PATH  = "C:/Users/jaish/Downloads/test (2).csv"
OUT_PATH   = "comp3/submission.csv"

SENSORS = ["TP2","TP3","H1","DV_pressure","Reservoirs","Oil_temperature",
           "Motor_current","COMP","DV_eletric","Towers","MPG","LPS",
           "Pressure_switch","Oil_level","Caudal_impulses"]


def build_features(df):
    """Time-based rolling features with closed=left to prevent leakage."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_ts = df.set_index("timestamp")

    # Cyclical time (no absolute date to avoid overfitting to April 17)
    df_ts["hour_sin"] = np.sin(2*np.pi*df_ts.index.hour/24)
    df_ts["hour_cos"] = np.cos(2*np.pi*df_ts.index.hour/24)
    df_ts["dow_sin"]  = np.sin(2*np.pi*df_ts.index.dayofweek/7)
    df_ts["dow_cos"]  = np.cos(2*np.pi*df_ts.index.dayofweek/7)

    # Gap indicator
    tdiffs = pd.Series(df_ts.index).diff()
    df_ts["is_gap_after"] = (tdiffs > pd.Timedelta("15min")).astype(int).values

    # Rolling stats at multiple windows (closed=left = no leakage)
    for w in ["1h","3h","6h","12h","24h"]:
        for col in SENSORS:
            r = df_ts[col].rolling(w, closed="left", min_periods=1)
            df_ts[f"{col}_std_{w}"]   = r.std().fillna(0)
            df_ts[f"{col}_mean_{w}"]  = r.mean()
            df_ts[f"{col}_range_{w}"] = r.max() - r.min()

    # Volatility ratio 1h/24h (drops to 0 on freeze)
    for col in SENSORS:
        s1  = df_ts[col].rolling("1h",  closed="left", min_periods=1).std().fillna(0)
        s24 = df_ts[col].rolling("24h", closed="left", min_periods=2).std().clip(lower=1e-8)
        df_ts[f"{col}_vol_ratio"] = (s1/s24).fillna(1.0)

    # Motor current frozen-at-idle features (the primary failure signal)
    mc = df_ts["Motor_current"]
    for w in ["30min","1h","3h","6h"]:
        mc_m = mc.rolling(w, closed="left", min_periods=1).mean()
        mc_s = mc.rolling(w, closed="left", min_periods=1).std().fillna(0)
        df_ts[f"mc_idle_mean_{w}"] = mc_m
        df_ts[f"mc_idle_std_{w}"]  = mc_s
        df_ts[f"mc_frozen_idle_{w}"] = ((mc_m < 0.1) & (mc_s < 0.01)).astype(float)

    # Count of consecutive idle windows (duration proxy)
    is_idle = (mc < 0.05).astype(float)
    for w in ["1h","3h","6h","12h"]:
        df_ts[f"mc_idle_count_{w}"] = is_idle.rolling(w, closed="left", min_periods=1).sum()

    # How many sensors are simultaneously frozen
    for w in ["1h","3h","6h"]:
        flags = [df_ts[f"{col}_std_{w}"] < 1e-4 for col in SENSORS]
        df_ts[f"n_frozen_{w}"] = sum(flags).astype(float)

    # Z-score from 24h baseline
    for col in ["TP2","TP3","Motor_current","Oil_temperature","Reservoirs"]:
        m24 = df_ts[col].rolling("24h", closed="left", min_periods=2).mean()
        s24 = df_ts[col].rolling("24h", closed="left", min_periods=2).std().clip(lower=1e-8)
        df_ts[f"{col}_zscore"] = ((df_ts[col]-m24)/s24).fillna(0)

    # Cross-sensor
    df_ts["pressure_gradient"] = df_ts["TP3"] - df_ts["TP2"]

    return df_ts.reset_index()


print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
train["_split"] = "train"
test["_split"]  = "test"
test["will_fail_next_6h"] = np.nan

combined = pd.concat([train, test], ignore_index=True)
combined["timestamp"] = pd.to_datetime(combined["timestamp"])
combined = combined.sort_values("timestamp").reset_index(drop=True)

print("Engineering features...")
combined = build_features(combined)

train_feat = combined[combined["_split"]=="train"].copy()
test_feat  = combined[combined["_split"]=="test"].copy()

FEAT_COLS = [c for c in train_feat.columns
             if c not in ["id","timestamp","will_fail_next_6h","_split"]]

X_all  = train_feat[FEAT_COLS].fillna(0).astype(np.float32)
y_all  = train_feat["will_fail_next_6h"].astype(int)
X_test = test_feat[FEAT_COLS].fillna(0).astype(np.float32)

n_pos = int(y_all.sum())
n_neg = int(len(y_all) - n_pos)
print(f"Train +:{n_pos} -:{n_neg}  features={len(FEAT_COLS)}")

# Model 1: Random Forest (balanced class weights, moderate depth)
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500, max_depth=7, min_samples_leaf=5,
    class_weight="balanced", n_jobs=-1, random_state=42, max_features="sqrt"
)
rf.fit(X_all, y_all)
rf_train = rf.predict_proba(X_all)[:,1]
rf_test  = rf.predict_proba(X_test)[:,1]

# Model 2: Logistic Regression (strong L2 regularization for generalization)
print("Training Logistic Regression...")
scaler   = StandardScaler()
X_all_s  = scaler.fit_transform(X_all)
X_test_s = scaler.transform(X_test)
lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000,
                        random_state=42, solver="lbfgs")
lr.fit(X_all_s, y_all)
lr_train = lr.predict_proba(X_all_s)[:,1]
lr_test  = lr.predict_proba(X_test_s)[:,1]

# Rule: motor frozen at idle for 3h
rule_train = train_feat["mc_frozen_idle_3h"].fillna(0).values.astype(float)
rule_test  = test_feat["mc_frozen_idle_3h"].fillna(0).values.astype(float)

# Ensemble (RF primary, LR secondary, rule as anchor)
train_probs = 0.45*rf_train + 0.30*lr_train + 0.25*rule_train
test_probs  = 0.45*rf_test  + 0.30*lr_test  + 0.25*rule_test

print(f"Train stats: max={train_probs.max():.4f}  Test stats: max={test_probs.max():.4f}")

# Threshold selection:
# At t=0.75: train_F1=0.974, test+=208 (best balance)
# At t=0.65: train_F1=0.961, test+=235 (more recall, more FP)
# We use 0.75: high training F1 and catches all major frozen events in test
THRESHOLD = 0.75
preds = (test_probs >= THRESHOLD).astype(int)

# Verify training performance
train_preds = (train_probs >= THRESHOLD).astype(int)
train_f1 = f1_score(y_all, train_preds, zero_division=0)
print(f"Threshold={THRESHOLD}  Train F1={train_f1:.4f}  Train+={train_preds.sum()}")
print(f"Test+={preds.sum()}/{len(preds)} ({100*preds.mean():.2f}%)")

# Feature importance
imp = pd.Series(rf.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
print("Top 15 RF feature importances:")
print(imp.head(15).to_string())

sub = pd.DataFrame({"id":test_feat["id"].astype(int), "will_fail_next_6h":preds})
sub = sub.sort_values("id").reset_index(drop=True)
sub.to_csv(OUT_PATH, index=False)
print(f"Saved {OUT_PATH}")
print(sub["will_fail_next_6h"].value_counts())
