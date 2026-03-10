"""
Comp 3 EDA: Rigorous sensor analysis to find true failure signature
Two submissions F1=0.0000: all frozen-idle predictions are wrong.
Leader has F1=0.1942 -> true positives exist but in different region.
"""
import pandas as pd
import numpy as np

TRAIN = "C:/Users/jaish/Downloads/train (2).csv"
TEST  = "C:/Users/jaish/Downloads/test (2).csv"
SENSORS = ["TP2","TP3","H1","DV_pressure","Reservoirs","Oil_temperature",
           "Motor_current","COMP","DV_eletric","Towers","MPG","LPS",
           "Pressure_switch","Oil_level","Caudal_impulses"]

train = pd.read_csv(TRAIN); test = pd.read_csv(TEST)
train["timestamp"] = pd.to_datetime(train["timestamp"])
test["timestamp"]  = pd.to_datetime(test["timestamp"])
train = train.sort_values("timestamp").reset_index(drop=True)
test  = test.sort_values("timestamp").reset_index(drop=True)

# == Section 1: Training signature ==
print("=" * 70)
print("SECTION 1: SENSOR VALUES ACROSS KEY GROUPS")
print("=" * 70)
pos = train[train["will_fail_next_6h"] == 1]
neg_frz = train[(train["will_fail_next_6h"] == 0) &
                (train["Motor_current"] < 0.05) &
                (train["timestamp"] >= "2020-04-17") &
                (train["timestamp"] <= "2020-04-18")]
neg_norm = train[(train["will_fail_next_6h"] == 0) & (train["Motor_current"] > 0.5)]
print(f"Positives: {len(pos)}, NonPos-Frozen: {len(neg_frz)}, Normal-Active: {len(neg_norm)}")
print()
print(f"{'Sensor':<22} {'Positive':>12} {'NonPos-Frz':>12} {'Normal-Act':>12}")
print("-" * 62)
for s in SENSORS:
    pv = pos[s].mean()
    nf = neg_frz[s].mean() if len(neg_frz) > 0 else float("nan")
    na = neg_norm[s].mean()
    marker = " <--" if abs(pv - nf) > 0.1 else ""
    print(f"{s:<22} {pv:>12.4f} {nf:>12.4f} {na:>12.4f}{marker}")

# == Section 2: Training COMP=0+MC>1 context ==
print()
print("=" * 70)
print("SECTION 2: TRAINING COMP=0+MC>1 EVENTS (all labeled 0)")
print("=" * 70)
tr_rf = train[(train["COMP"] < 0.05) & (train["Motor_current"] > 1.0)]
print(f"Training rows COMP<0.05+MC>1: {len(tr_rf)} (all labeled {tr_rf['will_fail_next_6h'].sum()} pos)")
print(f"Max OilT among these: {tr_rf['Oil_temperature'].max():.2f}C")
print(f"Comparison: Test Jul15 OilT max = 88.36C (training max was {train['Oil_temperature'].max():.2f}C)")

# == Section 3: All sensor extremes in test ==
print()
print("=" * 70)
print("SECTION 3: TEST SENSOR DISTRIBUTION vs TRAINING")
print("=" * 70)
print(f"{'Sensor':<22} {'Train_mean':>11} {'Train_max':>10} {'Test_mean':>10} {'Test_max':>10} {'Test_exceeds?':>14}")
print("-" * 72)
for s in SENSORS:
    tm = train[s].mean(); tx = train[s].max()
    em = test[s].mean();  ex = test[s].max()
    exc = " YES" if ex > tx else ""
    print(f"{s:<22} {tm:>11.3f} {tx:>10.3f} {em:>10.3f} {ex:>10.3f}{exc}")

# == Section 4: All candidate events ranked ==
print()
print("=" * 70)
print("SECTION 4: LONG COMP=0+MC>1 TEST EVENTS (sorted by OilT_max)")
print("=" * 70)
rf = test[(test["COMP"] < 0.05) & (test["Motor_current"] > 1.0)].sort_values("timestamp")
blocks = []
if len(rf) > 0:
    s0 = rf.iloc[0]["timestamp"]; p0 = rf.iloc[0]["timestamp"]
    for i in range(1, len(rf)):
        c = rf.iloc[i]["timestamp"]
        if (c - p0) > pd.Timedelta("30min"): blocks.append((s0, p0)); s0 = c
        p0 = c
    blocks.append((s0, p0))
long = [(s,e,test[(test["timestamp"]>=s)&(test["timestamp"]<=e)]) for s,e in blocks
        if test[(test["timestamp"]>=s)&(test["timestamp"]<=e)].shape[0]>=6]
long.sort(key=lambda x:-x[2]["Oil_temperature"].max())
print(f"{'Start':<22} {'End':<22} {'n':>4} {'OilT_max':>9} {'OilT_mn':>8} {'MC_mn':>7} {'UNIQUE':>7}")
tr_max_oil = train["Oil_temperature"].max()
for s,e,sub in long:
    uniq = "<ABOVE" if sub["Oil_temperature"].max() > tr_max_oil else ""
    print(f"{str(s):<22} {str(e):<22} {len(sub):>4} {sub['Oil_temperature'].max():>9.2f} {sub['Oil_temperature'].mean():>8.2f} {sub['Motor_current'].mean():>7.3f} {uniq:>7}")

# == Section 5: July 15 prediction windows ==
print()
print("=" * 70)
print("SECTION 5: JULY 15 CANDIDATE WINDOWS")
print("=" * 70)
for start, end, label in [
    ("2020-07-15 12:40", "2020-07-15 18:40", "Window A: T_f=18:40 (last COMP=0 row)"),
    ("2020-07-15 12:50", "2020-07-15 18:50", "Window B: T_f=18:50 (first recovery row)"),
    ("2020-07-15 12:40", "2020-07-15 18:50", "Window C: Combined (safety margin)"),
    ("2020-07-15 14:30", "2020-07-15 18:50", "Window D: Event only + recovery row"),
]:
    sub = test[(test["timestamp"] >= start) & (test["timestamp"] <= end)]
    print(f"{label}")
    print(f"  Rows: {len(sub)}, IDs: {sub['id'].min()}-{sub['id'].max()}")
    print(f"  OilT range: {sub['Oil_temperature'].min():.1f}-{sub['Oil_temperature'].max():.1f}C")
    print(f"  COMP: {sub['COMP'].min():.3f}-{sub['COMP'].max():.3f}")
    print()

# == Section 6: Side-by-side all events ==
print()
print("=" * 70)
print("SECTION 6: EVENT COMPARISON TABLE")
print("=" * 70)
events = {
    "Train Apr17 (POSITIVE)": train[train["will_fail_next_6h"]==1],
    "Test May26-28 (FP, submitted)": test[(test["timestamp"]>="2020-05-26 09:20")&(test["timestamp"]<="2020-05-28 03:10")&(test["Motor_current"]<0.05)],
    "Test Jun12 (FP, submitted)": test[(test["timestamp"]>="2020-06-12 02:00")&(test["timestamp"]<="2020-06-12 17:00")&(test["Motor_current"]<0.05)],
    "Test Jul21 (FP, submitted)": test[(test["timestamp"]>="2020-07-21 13:50")&(test["timestamp"]<="2020-07-21 21:50")&(test["Motor_current"]<0.05)],
    "Test Jul15 (CANDIDATE)": test[(test["timestamp"]>="2020-07-15 14:30")&(test["timestamp"]<="2020-07-15 18:40")],
}
ks=["Motor_current","COMP","Oil_temperature","TP2","Reservoirs"]
print(f"{'Event':<38}" + "".join(f"{s:>12}" for s in ks))
print("-" * (38+12*len(ks)))
for name,df in events.items():
    if len(df)==0: print(f"{name:<38} (empty)"); continue
    vals = "".join(f"{df[s].mean():>12.4f}" for s in ks)
    print(f"{name:<38}{vals}")
