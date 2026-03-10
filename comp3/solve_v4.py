"""
Comp 3 solve_v4: Anomaly-based prediction using Oil_temperature.

The training failure is frozen-idle (MC~0, COMP~0.79).
The test failure candidate (July 15) is a running failure (MC~5.5A, COMP~0, OilT~88C).
A supervised model trained on training features won't generalize to this failure mode,
so we use an anomaly approach:

1. OilT threshold: flag rows where OilT > training max (82.73C)
   - July 15 only: 26 rows during actual overheating (IDs 19724-19749)
2. 6h window before highest OilT row: capture run-up to failure
3. Rolling 6h max OilT: flag extended high-temp periods

Output: submission_v4.csv with anomaly-based predictions.
Compare with solve_v3 (rule-based) to pick best submission.
"""
import pandas as pd
import numpy as np

TRAIN_PATH = "C:/Users/jaish/Downloads/train (2).csv"
TEST_PATH  = "C:/Users/jaish/Downloads/test (2).csv"
OUT_PATH   = "comp3/submission_v4.csv"

print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
train["timestamp"] = pd.to_datetime(train["timestamp"])
test["timestamp"]  = pd.to_datetime(test["timestamp"])
train = train.sort_values("timestamp").reset_index(drop=True)
test  = test.sort_values("timestamp").reset_index(drop=True)

TRAIN_MAX_OILT  = train["Oil_temperature"].max()
TRAIN_MEAN_OILT = train["Oil_temperature"].mean()
TRAIN_STD_OILT  = train["Oil_temperature"].std()
print("Training OilT: mean=%.2f std=%.2f max=%.2f" % (TRAIN_MEAN_OILT, TRAIN_STD_OILT, TRAIN_MAX_OILT))

# Compute rolling 6h max OilT in test (window = 36 rows at 10min intervals)
WINDOW = 37  # 6h + 1 (inclusive of current row)
test = test.sort_values("timestamp").reset_index(drop=True)
test["oilt_roll_max_6h"]  = test["Oil_temperature"].rolling(WINDOW, min_periods=1).max()
test["oilt_roll_mean_6h"] = test["Oil_temperature"].rolling(WINDOW, min_periods=1).mean()

# Anomaly score: how far above training max is the 6h rolling max?
test["oilt_anomaly"] = (test["oilt_roll_max_6h"] - TRAIN_MAX_OILT).clip(lower=0)

# Also flag running-failure signature: COMP<0.05 AND MC>3
test["running_fail"] = ((test["COMP"] < 0.05) & (test["Motor_current"] > 3.0)).astype(int)
test["rf_roll_6h"]   = test["running_fail"].rolling(WINDOW, min_periods=1).sum()

# Combined anomaly score (high OilT AND running failure signature)
test["combined_score"] = test["oilt_anomaly"] * (test["rf_roll_6h"] / WINDOW + 0.01)

print()
print("--- Approach 1: OilT > training max directly ---")
mask1 = test["Oil_temperature"] > TRAIN_MAX_OILT
print("Rows with OilT > %.2f: %d" % (TRAIN_MAX_OILT, mask1.sum()))
if mask1.sum() > 0:
    ids1 = test.loc[mask1, "id"]
    print("IDs: %d to %d" % (ids1.min(), ids1.max()))
    print("Timestamps: %s to %s" % (test.loc[mask1,"timestamp"].min(), test.loc[mask1,"timestamp"].max()))

print()
print("--- Approach 2: 6h window before last high-OilT row ---")
# Find the row with peak OilT in test
peak_idx = test["Oil_temperature"].idxmax()
peak_ts  = test.loc[peak_idx, "timestamp"]
peak_oilt= test.loc[peak_idx, "Oil_temperature"]
print("Peak OilT: %.2f at %s (ID %d)" % (peak_oilt, peak_ts, test.loc[peak_idx, "id"]))
win_start2 = peak_ts - pd.Timedelta(hours=6) + pd.Timedelta(minutes=10)
win_end2   = peak_ts
mask2 = (test["timestamp"] >= win_start2) & (test["timestamp"] <= win_end2)
ids2  = test.loc[mask2, "id"]
print("6h window: %s to %s" % (win_start2, win_end2))
print("Rows: %d, IDs %d-%d" % (mask2.sum(), ids2.min(), ids2.max()))

print()
print("--- Approach 3: Combined anomaly score threshold ---")
# Find the 37-row window with highest combined score
best_score = -1
best_end_idx = None
for i in range(WINDOW-1, len(test)):
    window_score = test.loc[i-WINDOW+1:i, "combined_score"].sum()
    if window_score > best_score:
        best_score = window_score
        best_end_idx = i

if best_end_idx is not None:
    start_idx3 = best_end_idx - WINDOW + 1
    end_idx3   = best_end_idx
    ts3_start  = test.loc[start_idx3, "timestamp"]
    ts3_end    = test.loc[end_idx3, "timestamp"]
    ids3 = test.loc[start_idx3:end_idx3, "id"]
    print("Best 37-row window by combined score: %s to %s" % (ts3_start, ts3_end))
    print("IDs: %d to %d, score=%.4f" % (ids3.min(), ids3.max(), best_score))
    oilt3 = test.loc[start_idx3:end_idx3, "Oil_temperature"]
    print("OilT: min=%.2f mean=%.2f max=%.2f" % (oilt3.min(), oilt3.mean(), oilt3.max()))

print()
print("--- Approach 4: Top-N by combined score (n=37 rows) ---")
top37 = test.nlargest(37, "combined_score")
ids4  = top37["id"]
oilt4 = top37["Oil_temperature"]
print("Top 37 rows by combined score:")
print("IDs min=%d max=%d" % (ids4.min(), ids4.max()))
print("OilT: min=%.2f mean=%.2f max=%.2f" % (oilt4.min(), oilt4.mean(), oilt4.max()))
ts4 = top37["timestamp"]
print("Timestamps: %s to %s" % (ts4.min(), ts4.max()))

print()
print("--- Decision: Use Approach 2 (6h before peak OilT) ---")
print("Primary (solve_v3): rule-based 12:40-18:40 July 15 (37 rows)")
print("Backup  (solve_v4): 6h window before peak OilT moment")

# Use approach 2 for submission_v4
preds = mask2.astype(int)
sub = pd.DataFrame({"id": test["id"].astype(int), "will_fail_next_6h": preds})
sub = sub.sort_values("id").reset_index(drop=True)
sub.to_csv(OUT_PATH, index=False)
print()
print("Saved %s" % OUT_PATH)
print(sub["will_fail_next_6h"].value_counts().to_string())

print()
print("--- Comparison summary ---")
print("solve_v3 (primary): 37 rows, 12:40-18:40, IDs 19713-19749")
print("solve_v4 (backup):  %d rows, %s to %s, IDs %d-%d" % (
    mask2.sum(), win_start2, win_end2, ids2.min(), ids2.max()))
