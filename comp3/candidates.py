"""
Comp 3: Enumerate ALL candidate failure windows in test data.
Ranks COMP<0.05+MC>1 events by Oil_temperature severity.
The failure candidate has the highest OilT (above training max).
"""
import pandas as pd
import numpy as np

TRAIN_PATH = "C:/Users/jaish/Downloads/train (2).csv"
TEST_PATH  = "C:/Users/jaish/Downloads/test (2).csv"

print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
train["timestamp"] = pd.to_datetime(train["timestamp"])
test["timestamp"]  = pd.to_datetime(test["timestamp"])
train = train.sort_values("timestamp").reset_index(drop=True)
test  = test.sort_values("timestamp").reset_index(drop=True)

TRAIN_MAX_OILT = train["Oil_temperature"].max()
TRAIN_MAX_MC   = train["Motor_current"].max()
print("Training sensor maxima: OilT=%.2f, MC=%.2f" % (TRAIN_MAX_OILT, TRAIN_MAX_MC))

# Find all COMP<0.05 + MC>1 events in TEST
test["rf"] = (test["COMP"] < 0.05) & (test["Motor_current"] > 1.0)
blk = (test["rf"] != test["rf"].shift()).cumsum()
test["block"] = blk

events = []
for blk_id, grp in test[test["rf"]].groupby("block"):
    dur_h = len(grp) * 10 / 60
    events.append({
        "ts_start":  grp["timestamp"].min(),
        "ts_end":    grp["timestamp"].max(),
        "n_rows":    len(grp),
        "dur_h":     round(dur_h, 2),
        "OilT_max":  round(grp["Oil_temperature"].max(), 2),
        "OilT_mean": round(grp["Oil_temperature"].mean(), 2),
        "MC_mean":   round(grp["Motor_current"].mean(), 2),
        "COMP_mean": round(grp["COMP"].mean(), 3),
        "id_min":    grp["id"].min(),
        "id_max":    grp["id"].max(),
        "exceeds":   grp["Oil_temperature"].max() > TRAIN_MAX_OILT,
    })

events_df = pd.DataFrame(events).sort_values("OilT_max", ascending=False).reset_index(drop=True)
print()
print("--- All COMP<0.05+MC>1 events in TEST ranked by OilT_max ---")
print(events_df.to_string(index=False))

# Candidate windows: last 6h before each event ends
print()
print("--- Candidate 6h windows (last 6h of each event) ---")
candidates = []
for _, ev in events_df.iterrows():
    win_end   = ev["ts_end"]
    win_start = win_end - pd.Timedelta(hours=6) + pd.Timedelta(minutes=10)
    mask = (test["timestamp"] >= win_start) & (test["timestamp"] <= win_end)
    n    = mask.sum()
    ids  = test.loc[mask, "id"]
    candidates.append({
        "event_start": ev["ts_start"],
        "event_end":   ev["ts_end"],
        "dur_h":       ev["dur_h"],
        "OilT_max":    ev["OilT_max"],
        "exceeds":     ev["exceeds"],
        "win_start":   win_start,
        "win_end":     win_end,
        "n_pos":       n,
        "id_min":      ids.min() if n > 0 else None,
        "id_max":      ids.max() if n > 0 else None,
    })

cand_df = pd.DataFrame(candidates)
print(cand_df.to_string(index=False))

# Detail on top candidate
top = events_df.iloc[0]
print()
print("--- TOP CANDIDATE: %s to %s ---" % (top["ts_start"], top["ts_end"]))
print("  dur=%.2fh OilT_max=%.2fC MC=%.2fA COMP=%.3f" % (
    top["dur_h"], top["OilT_max"], top["MC_mean"], top["COMP_mean"]))
status = "EXCEEDS" if top["exceeds"] else "within"
print("  OilT %s training max (%.2fC)" % (status, TRAIN_MAX_OILT))

event_end = top["ts_end"]
ws_a = event_end - pd.Timedelta(hours=6) + pd.Timedelta(minutes=10)
we_a = event_end
ws_b = event_end - pd.Timedelta(hours=6)
we_b = event_end - pd.Timedelta(minutes=10)

for label, ws, we in [
    ("Window A (6h end-aligned)", ws_a, we_a),
    ("Window B (6h shift -10min)", ws_b, we_b),
]:
    mask = (test["timestamp"] >= ws) & (test["timestamp"] <= we)
    ids  = test.loc[mask, "id"]
    oilt = test.loc[mask, "Oil_temperature"]
    print()
    print("  %s: %s to %s" % (label, ws, we))
    print("    n_rows=%d, IDs %d-%d" % (mask.sum(), ids.min(), ids.max()))
    print("    OilT: min=%.2f mean=%.2f max=%.2f" % (oilt.min(), oilt.mean(), oilt.max()))

mask_c = (test["timestamp"] >= top["ts_start"]) & (test["timestamp"] <= top["ts_end"])
ids_c  = test.loc[mask_c, "id"]
print()
print("  Window C (full event): %s to %s" % (top["ts_start"], top["ts_end"]))
print("    n_rows=%d, IDs %d-%d" % (mask_c.sum(), ids_c.min(), ids_c.max()))

print()
print("--- Row-level detail for top event ---")
mask_ev = (test["timestamp"] >= top["ts_start"]) & (test["timestamp"] <= top["ts_end"])
cols = ["id", "timestamp", "Motor_current", "COMP", "Oil_temperature", "TP2", "DV_pressure"]
print(test.loc[mask_ev, cols].to_string(index=False))
print()
print("Done.")
