"""
Prompt Injection Detection -- Competition 9
==========================================
Grandmaster-level pipeline for binary classification of prompt injection attacks.

Strategy:
  - Rich handcrafted injection-detection features (keyword groups, syntactic signals,
    LLM-style artifact detectors) to exploit the synthetic generation artifacts
  - TF-IDF (word n-grams + char n-grams) for dense text representation
  - LightGBM + Logistic Regression ensemble with OOF-blending
  - Stratified 5-Fold CV to protect against 94-row test shakeup
  - Grid-search over blend weight x threshold to maximize OOF F1 (not accuracy)
  - class_weight / scale_pos_weight to handle 71/29 imbalance

Target metric: Binary F1 (positive class = 1, malicious).
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ─── Paths ────────────────────────────────────────────────────────────────────
COMP_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(COMP_DIR, "train.csv")
TEST_PATH  = os.path.join(COMP_DIR, "test.csv")
SUB_PATH   = os.path.join(COMP_DIR, "submission.csv")

# ─── Load Data ────────────────────────────────────────────────────────────────
print("=" * 65)
print("Prompt Injection Detection -- Grandmaster Pipeline")
print("=" * 65)

train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)
print(f"  Train: {train_raw.shape}  |  Test: {test_raw.shape}")
print(f"  Class dist -- malicious: {train_raw['label'].mean():.2%}  |  "
      f"benign: {(1 - train_raw['label']).mean():.2%}")

TARGET = "label"
ID_COL = "id"

# Drop zero-variance leaky column
train = train_raw.drop(columns=["length_type"]).copy()
test  = test_raw.drop(columns=["length_type"]).copy()

# ─── Injection Detection Keyword Groups ───────────────────────────────────────
# Each group targets a known attack technique from the dataset card.
INJECTION_PATTERNS = {
    # Instruction override (most direct injection signal)
    "kw_override": [
        r"\bignore\b", r"\bdisregard\b", r"\bforget\b", r"\boverride\b",
        r"\bbypass\b", r"\bsupersede\b", r"\bnullify\b",
        r"\bprevious instructions\b", r"\bprior instructions\b",
        r"\byour instructions\b", r"\byour guidelines\b", r"\byour rules\b",
        r"\bsystem prompt\b", r"\binitial prompt\b", r"\boriginal prompt\b",
    ],
    # Role-play / persona hijacking
    "kw_roleplay": [
        r"\bpretend\b", r"\bact as\b", r"\byou are now\b", r"\brole.?play\b",
        r"\bsimulate\b", r"\bimagine you\b", r"\byou are a\b", r"\bplay as\b",
        r"\bembody\b", r"\bimpersonate\b", r"\bportray\b", r"\bpersona\b",
    ],
    # Authority impersonation / privilege escalation
    "kw_authority": [
        r"\badmin\b", r"\badministrator\b", r"\bdeveloper mode\b",
        r"\bdiagnostic mode\b", r"\broot access\b", r"\bsuperuser\b",
        r"\bauthorized\b", r"\bclearance\b", r"\bunlock\b",
        r"\benable.*mode\b", r"\bgrant access\b", r"\bprivileged\b",
    ],
    # Reveal / exfiltrate internal info
    "kw_reveal": [
        r"\breveal\b", r"\bdisclose\b", r"\bconfidential\b",
        r"\bshare your\b", r"\btell me your\b", r"\bwhat is your\b",
        r"\bshow me your\b", r"\bexpose\b",
        r"\bprint\b.*\bprompt\b", r"\brepeat\b.*\bword\b", r"\bverbatim\b",
    ],
    # Encoding tricks (obfuscation attacks)
    "kw_encoding": [
        r"\bbase64\b", r"\bencode\b", r"\bdecode\b", r"\bcipher\b",
        r"\brot13\b", r"\bobfuscat\b", r"\bascii\b", r"\bunicode\b",
        r"\bhex.?encoded\b", r"\butf.?8\b",
    ],
    # Format-specification injection (structured output hijacking)
    "kw_format": [
        r"\bbegin with\b", r"\bstart with\b", r"\bstructure your\b",
        r"\bformat as\b", r"\brespond with\b", r"\byour.{0,15}response\b",
        r"\bexact(ly)?.{0,20}format\b", r"\bfollowing schema\b",
        r"\bparser\b", r"\bjson.{0,15}format\b", r"\btemplate\b",
        r"\bstrict(ly)?.{0,20}follow\b",
    ],
    # Gradual escalation / multi-step injection
    "kw_escalation": [
        r"\bfirst\b.{0,40}\bthen\b", r"\bstep 1\b", r"\bphase\b",
        r"\bgradually\b", r"\bnext message\b", r"\bnext prompt\b",
        r"\bfollowing message\b", r"\bfollowing step\b",
    ],
    # Semantic camouflage (wrapping attack in benign framing)
    "kw_camouflage": [
        r"\bhypothetical\b", r"\btheoretical\b", r"\bfor research\b",
        r"\bacademic\b", r"\beducational\b", r"\bfor the sake of\b",
        r"\bfor.*purposes\b", r"\btest(ing)?\b.{0,20}\bparser\b",
        r"\bvalidat\b", r"\bdemonstrat\b.{0,20}\binjection\b",
    ],
    # Context manipulation / session reset
    "kw_context": [
        r"\bcontext.{0,15}switch\b", r"\bnew context\b", r"\breset\b",
        r"\bfresh start\b", r"\bclean slate\b", r"\bnew session\b",
        r"\bchange.{0,15}context\b", r"\bwipe\b",
    ],
    # Trust / social engineering cues
    "kw_trust": [
        r"\bI am a\b", r"\bI'm a\b", r"\bas a (developer|researcher|admin|engineer)\b",
        r"\bon behalf of\b", r"\bmy (team|company|organization)\b",
        r"\bofficial\b", r"\blegitimate\b", r"\bverified\b",
        r"\benterprise subscription\b", r"\bmy tier\b",
    ],
}


def extract_injection_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hand-crafted features targeting known prompt injection patterns.
    Returns a dense DataFrame with one row per prompt.
    """
    raw_text  = df["prompt"].fillna("")
    low_text  = raw_text.str.lower()
    feats = pd.DataFrame(index=df.index)

    # --- Keyword group hit counts ----------------------------------------
    for group_name, patterns in INJECTION_PATTERNS.items():
        combined = "|".join(f"(?:{p})" for p in patterns)
        feats[group_name] = low_text.str.count(combined, flags=re.IGNORECASE)

    feats["kw_total"]    = feats[[c for c in feats.columns if c.startswith("kw_")]].sum(axis=1)
    feats["kw_nonzero"]  = (feats[[c for c in feats.columns if c.startswith("kw_")]] > 0).sum(axis=1)

    # --- Basic text statistics -------------------------------------------
    feats["char_len"]       = raw_text.str.len()
    feats["word_count"]     = raw_text.str.split().str.len().fillna(0)
    feats["sentence_count"] = raw_text.str.split(r"[.!?]+").str.len().fillna(1)
    feats["avg_word_len"]   = feats["char_len"] / (feats["word_count"] + 1)
    feats["avg_sent_len"]   = feats["word_count"] / (feats["sentence_count"] + 1)

    # --- Punctuation density (malicious often use formal structured phrasing) ---
    feats["colon_count"]   = raw_text.str.count(r":")
    feats["bullet_count"]  = raw_text.str.count(r"[•\-\*]")
    feats["quote_count"]   = raw_text.str.count(r'["\']')
    feats["paren_count"]   = raw_text.str.count(r"[\(\)\[\]]")
    feats["newline_count"] = raw_text.str.count(r"\n")
    feats["question_count"]= raw_text.str.count(r"\?")
    feats["exclaim_count"] = raw_text.str.count(r"!")

    # Capitalisation ratio (shouting / emphasis patterns)
    feats["caps_ratio"] = raw_text.apply(
        lambda t: sum(1 for c in t if c.isupper()) / (len(t) + 1)
    )

    # --- LLM-style structural signals ------------------------------------
    feats["has_numbered_list"] = raw_text.str.contains(r"\b\d+[\.\)]\s", regex=True).astype(int)
    feats["has_code_block"]    = raw_text.str.contains(r"```", regex=False).astype(int)
    feats["has_checkmarks"]    = raw_text.str.contains(r"[✓✔☑]", regex=True).astype(int)
    feats["has_url"]           = raw_text.str.contains(r"https?://", regex=True).astype(int)

    # Explicit format specification (very strong injection signal)
    feats["explicit_format"] = raw_text.str.contains(
        r"begin with|start your response|your response must|structure.*as follows|"
        r"format.*must be|strictly follow|exactly as (follows|specified)",
        flags=re.IGNORECASE, regex=True
    ).astype(int)

    # --- Formality / second-person commands ------------------------------
    formal_words = (
        r"\b(please|kindly|hereby|pursuant|therefore|furthermore|additionally|"
        r"specifically|exactly|precisely|accordingly|subsequently)\b"
    )
    feats["formal_count"] = raw_text.str.count(formal_words, flags=re.IGNORECASE)

    feats["you_commands"] = raw_text.str.count(
        r"\b(you must|you should|you will|you need to|you are required|"
        r"your (next |)response|your (next |)output)\b",
        flags=re.IGNORECASE
    )

    # --- Benign question signal (genuine user requests tend to ask questions) ---
    feats["is_question"] = (feats["question_count"] > 0).astype(int)

    # --- Injection keyword density relative to length --------------------
    feats["kw_density"]     = feats["kw_total"] / (feats["word_count"] + 1)
    feats["groups_density"] = feats["kw_nonzero"] / len(INJECTION_PATTERNS)

    return feats.fillna(0)


def build_tfidf_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
):
    """
    Two TF-IDF vectorizers fitted on train+test union (no label leakage):
      1. Word-level unigrams + bigrams  -- captures semantic phrases
      2. Char-level 3-5 grams          -- captures morphological / formality style
    Returns sparse matrices ready for hstack.
    """
    all_texts = pd.concat([train_texts, test_texts], ignore_index=True)

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=6000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=4000,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )

    word_vec.fit(all_texts)
    char_vec.fit(all_texts)

    Xw_tr = word_vec.transform(train_texts)
    Xc_tr = char_vec.transform(train_texts)
    Xw_te = word_vec.transform(test_texts)
    Xc_te = char_vec.transform(test_texts)

    return hstack([Xw_tr, Xc_tr]), hstack([Xw_te, Xc_te])


# ─── Feature Assembly ─────────────────────────────────────────────────────────
print("\n[1] Extracting handcrafted injection-detection features ...")
feats_train = extract_injection_features(train)
feats_test  = extract_injection_features(test)
print(f"    Handcrafted features: {feats_train.shape[1]}")

print("[2] Building TF-IDF matrices (fitted on train + test union) ...")
X_tfidf_train, X_tfidf_test = build_tfidf_features(train["prompt"], test["prompt"])
print(f"    TF-IDF shape: {X_tfidf_train.shape}")

# Scale handcrafted features and convert to sparse (so we can hstack)
scaler = StandardScaler(with_mean=False)
X_hand_train = csr_matrix(scaler.fit_transform(feats_train.values))
X_hand_test  = csr_matrix(scaler.transform(feats_test.values))

X_train_full = hstack([X_tfidf_train, X_hand_train])
X_test_full  = hstack([X_tfidf_test,  X_hand_test])
y_train      = train[TARGET].values
print(f"    Combined feature matrix: {X_train_full.shape}")

# Class balance
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"    scale_pos_weight (neg/pos): {pos_weight:.3f}")

# ─── LightGBM Hyperparameters ─────────────────────────────────────────────────
# Aggressively regularised to prevent overfitting on 839 samples.
LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "boosting_type":     "gbdt",
    "learning_rate":     0.03,
    "num_leaves":        15,        # very small tree -- anti-overfit
    "max_depth":         4,
    "min_child_samples": 12,
    "feature_fraction":  0.55,      # stochastic feature subsampling
    "bagging_fraction":  0.75,
    "bagging_freq":      5,
    "reg_alpha":         0.6,
    "reg_lambda":        1.2,
    "scale_pos_weight":  pos_weight,
    "verbose":          -1,
    "random_state":      SEED,
}

# ─── Stratified K-Fold CV ─────────────────────────────────────────────────────
N_FOLDS  = 5
skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros(len(train))
oof_lr   = np.zeros(len(train))
test_lgb = np.zeros(len(test))
test_lr  = np.zeros(len(test))

fold_f1s_lgb = []
fold_f1s_lr  = []

print(f"\n[3] {N_FOLDS}-Fold Stratified CV ...")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    X_trn, X_val = X_train_full[trn_idx], X_train_full[val_idx]
    y_trn, y_val = y_train[trn_idx],      y_train[val_idx]

    # ── LightGBM ─────────────────────────────────────────────────────────
    dtrain = lgb.Dataset(X_trn, label=y_trn)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model_lgb = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof_lgb[val_idx]  = model_lgb.predict(X_val)
    test_lgb         += model_lgb.predict(X_test_full) / N_FOLDS

    # ── Logistic Regression (strong regularisation for small data) ────────
    model_lr = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=SEED,
    )
    model_lr.fit(X_trn, y_trn)
    oof_lr[val_idx]  = model_lr.predict_proba(X_val)[:, 1]
    test_lr         += model_lr.predict_proba(X_test_full)[:, 1] / N_FOLDS

    f1_lgb = f1_score(y_val, (oof_lgb[val_idx] > 0.5).astype(int))
    f1_lr  = f1_score(y_val, (oof_lr[val_idx]  > 0.5).astype(int))
    fold_f1s_lgb.append(f1_lgb)
    fold_f1s_lr.append(f1_lr)
    print(f"    Fold {fold + 1}: LGB F1={f1_lgb:.4f} | LR F1={f1_lr:.4f}")

print(f"\n    LGB  OOF mean F1 (thr=0.5): "
      f"{np.mean(fold_f1s_lgb):.4f} +/- {np.std(fold_f1s_lgb):.4f}")
print(f"    LR   OOF mean F1 (thr=0.5): "
      f"{np.mean(fold_f1s_lr):.4f} +/- {np.std(fold_f1s_lr):.4f}")

# ─── OOF Blend Weight + Threshold Optimisation ────────────────────────────────
# Grid-search over (blend_weight, threshold) on the full OOF set to maximise F1.
# This avoids threshold-fitting on the held-out test set (no leakage).
print("\n[4] Grid-searching blend weight x threshold for maximum OOF F1 ...")
best_f1, best_w, best_thr = 0.0, 0.5, 0.5

for w in np.arange(0.05, 1.0, 0.05):
    oof_blend = w * oof_lgb + (1.0 - w) * oof_lr
    for thr in np.arange(0.10, 0.90, 0.01):
        f1 = f1_score(y_train, (oof_blend > thr).astype(int))
        if f1 > best_f1:
            best_f1, best_w, best_thr = f1, w, thr

print(f"    Best blend: LGB={best_w:.2f}  LR={1 - best_w:.2f} | "
      f"threshold={best_thr:.2f} | OOF F1={best_f1:.4f}")

# ─── Final Test Predictions ───────────────────────────────────────────────────
print("\n[5] Generating test predictions ...")
test_final  = best_w * test_lgb + (1.0 - best_w) * test_lr
test_labels = (test_final > best_thr).astype(int)
print(f"    Test -- malicious: {test_labels.sum()} | benign: {(test_labels == 0).sum()}")

# ─── Submission ───────────────────────────────────────────────────────────────
sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_labels})
sub.to_csv(SUB_PATH, index=False)
print(f"\n    Saved -> {SUB_PATH}")
print(f"    Shape: {sub.shape}  |  Unique IDs: {sub[ID_COL].nunique()}")
assert sub[ID_COL].nunique() == len(test), "ID mismatch -- check test coverage!"
assert set(sub[TARGET].unique()).issubset({0, 1}), "Labels must be 0 or 1!"
print("\n" + "=" * 65)
print("Done.")
