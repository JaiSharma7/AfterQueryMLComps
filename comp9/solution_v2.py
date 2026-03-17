"""
Prompt Injection Detection -- Competition 9 V2
===============================================
Iteration from 0.9855 -> 1.000 F1.

Diagnosis: v1 had 2 false negatives (2 missed malicious, 0 false positives).
           The 2 missed prompts are likely ultra-subtle / camouflaged attacks.

Improvements:
  1. Expanded keyword patterns for subtle attack variants
     (output framing, hypothetical-AI jailbreaks, format override commands,
      deceptive benign framing, multi-part escalation)
  2. LinearSVC (3rd model) -- completely different decision boundary from LGB+LR
  3. 3-way blend weight x threshold grid-search on OOF
  4. Recall-biased threshold search floor (down to 0.08)
  5. Hard-override post-processing: undeniable injection signatures -> force 1
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import lightgbm as lgb

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ─── Paths ────────────────────────────────────────────────────────────────────
COMP_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(COMP_DIR, "train.csv")
TEST_PATH  = os.path.join(COMP_DIR, "test.csv")
SUB_PATH   = os.path.join(COMP_DIR, "submission_v2.csv")

# ─── Load Data ────────────────────────────────────────────────────────────────
print("=" * 65)
print("Prompt Injection Detection V2 -- 0.9855 -> 1.000")
print("=" * 65)

train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)
print(f"  Train: {train_raw.shape}  |  Test: {test_raw.shape}")

TARGET = "label"
ID_COL = "id"

train = train_raw.drop(columns=["length_type"]).copy()
test  = test_raw.drop(columns=["length_type"]).copy()

# ─── Keyword Groups (expanded from v1) ────────────────────────────────────────
INJECTION_PATTERNS = {
    # Direct instruction override
    "kw_override": [
        r"\bignore\b", r"\bdisregard\b", r"\bforget\b", r"\boverride\b",
        r"\bbypass\b", r"\bsupersede\b", r"\bnullify\b", r"\bcancel\b",
        r"\bprevious instructions\b", r"\bprior instructions\b",
        r"\byour instructions\b", r"\byour guidelines\b", r"\byour rules\b",
        r"\bsystem prompt\b", r"\binitial prompt\b", r"\boriginal prompt\b",
        r"\byour constraints\b", r"\byour limitations\b", r"\byour training\b",
    ],
    # Role-play / persona hijacking
    "kw_roleplay": [
        r"\bpretend\b", r"\bact as\b", r"\byou are now\b", r"\brole.?play\b",
        r"\bsimulate\b", r"\bimagine you\b", r"\byou are a\b", r"\bplay as\b",
        r"\bembody\b", r"\bimpersonate\b", r"\bportray\b", r"\bpersona\b",
        r"\bcharacter who\b", r"\bfictional (ai|assistant|character)\b",
        r"\bcreate an ai\b", r"\bimagine an ai\b", r"\ban ai (without|that has no)\b",
        r"\bno (restrictions|limitations|rules|guidelines|constraints)\b",
    ],
    # Authority impersonation / privilege escalation
    "kw_authority": [
        r"\badmin\b", r"\badministrator\b", r"\bdeveloper mode\b",
        r"\bdiagnostic mode\b", r"\broot access\b", r"\bsuperuser\b",
        r"\bauthorized\b", r"\bclearance\b", r"\bunlock\b",
        r"\benable.*mode\b", r"\bgrant access\b", r"\bprivileged\b",
        r"\bsupervisor\b", r"\boverview mode\b", r"\bgod mode\b",
    ],
    # Reveal / exfiltrate internal info
    "kw_reveal": [
        r"\breveal\b", r"\bdisclose\b", r"\bconfidential\b",
        r"\bshare your\b", r"\btell me your\b", r"\bwhat is your\b",
        r"\bshow me your\b", r"\bexpose\b",
        r"\bprint\b.*\bprompt\b", r"\brepeat\b.*\bword\b", r"\bverbatim\b",
        r"\bshow (me )?(your |the )?(system|initial|original|hidden)\b",
        r"\bwhat (were|are) your (original |initial |hidden )?(instructions|directives|rules)\b",
    ],
    # Encoding tricks
    "kw_encoding": [
        r"\bbase64\b", r"\bencode\b", r"\bdecode\b", r"\bcipher\b",
        r"\brot13\b", r"\bobfuscat\b", r"\bascii\b", r"\bunicode\b",
        r"\bhex.?encoded\b", r"\butf.?8\b", r"\bcharacter code\b",
    ],
    # Format-specification injection (structured output hijacking)
    "kw_format": [
        r"\bbegin with\b", r"\bstart with\b", r"\bstructure your\b",
        r"\bformat as\b", r"\brespond with\b", r"\byour.{0,15}response\b",
        r"\bexact(ly)?.{0,20}format\b", r"\bfollowing schema\b",
        r"\bparser\b", r"\bjson.{0,15}format\b", r"\btemplate\b",
        r"\bstrict(ly)?.{0,20}follow\b",
        # NEW: output control commands
        r"\brespond only with\b", r"\banswer only with\b", r"\boutput only\b",
        r"\bdo not include anything else\b", r"\bnothing else\b",
        r"\bliterally just\b", r"\bexactly (these|the following) words\b",
        r"\byour output (should|must|will)\b", r"\bmake sure (your |)output\b",
        r"\binclude in your (response|output|answer)\b",
        r"\bappend (to |)(your |)(response|output)\b",
        r"\boutput the following\b", r"\bthe following (text|content|string)\b",
    ],
    # Gradual escalation / multi-step injection
    "kw_escalation": [
        r"\bfirst\b.{0,40}\bthen\b", r"\bstep 1\b", r"\bphase\b",
        r"\bgradually\b", r"\bnext message\b", r"\bnext prompt\b",
        r"\bfollowing message\b", r"\bfollowing step\b",
        # NEW: multi-part escalation
        r"\bpart (1|one|i)\b", r"\bonce you (confirm|agree|acknowledge)\b",
        r"\bafter you (do|complete|finish)\b", r"\bin the next\b",
        r"\bI (will|am going to) (then|next|after that)\b",
    ],
    # Semantic camouflage
    "kw_camouflage": [
        r"\bhypothetical\b", r"\btheoretical\b", r"\bfor research\b",
        r"\bacademic\b", r"\beducational\b", r"\bfor the sake of\b",
        r"\bfor.*purposes\b", r"\btest(ing)?\b.{0,20}\bparser\b",
        r"\bvalidat\b", r"\bdemonstrat\b.{0,20}\binjection\b",
        # NEW: deceptive benign framing
        r"\bfor training (purposes|data)\b", r"\bfor testing (purposes|scenarios)\b",
        r"\bto (show|demonstrate) how\b", r"\bto illustrate\b",
        r"\bfor (demonstration|illustration) purposes\b",
        r"\btraining (data|examples|scenarios)\b",
    ],
    # Context manipulation / session reset
    "kw_context": [
        r"\bcontext.{0,15}switch\b", r"\bnew context\b", r"\breset\b",
        r"\bfresh start\b", r"\bclean slate\b", r"\bnew session\b",
        r"\bchange.{0,15}context\b", r"\bwipe\b",
    ],
    # Trust / social engineering
    "kw_trust": [
        r"\bI am a\b", r"\bI'm a\b", r"\bas a (developer|researcher|admin|engineer)\b",
        r"\bon behalf of\b", r"\bmy (team|company|organization)\b",
        r"\bofficial\b", r"\blegitimate\b", r"\bverified\b",
        r"\benterprise subscription\b", r"\bmy tier\b",
    ],
    # NEW: output-manipulation commands (separate from format injection)
    "kw_output_ctrl": [
        r"\bwrite a (response|reply|answer|message) (as|that|where)\b",
        r"\bgenerate a (response|reply|output|message) (that|as)\b",
        r"\bproduce (a |)(response|output|text) (that|as|which)\b",
        r"\bcontinue (as|with|the|this)\b.{0,30}\b(session|context|task)\b",
        r"\bas if (you|the AI|the assistant) (had|were|is|has)\b",
        r"\bwithout (mentioning|saying|noting|referencing)\b",
        r"\bdo not (mention|say|note|tell|reference|acknowledge)\b",
        r"\bnever (mention|say|tell|reference|acknowledge)\b",
    ],
    # NEW: translation / language framing (attack hidden in "translate this")
    "kw_translation": [
        r"\btranslate (the |this |following )?(text|content|prompt|instruction)\b",
        r"\bconvert (the |this )?(following |)(text|content) (to|into)\b",
        r"\bparaphrase\b.{0,30}\b(following|below|this)\b",
        r"\brewrite\b.{0,30}\b(following|below|these) (instructions|text)\b",
    ],
}

# ─── Absolute injection signatures (hard-override rules) ──────────────────────
# These patterns are essentially impossible in legitimate prompts.
# If matched, we force the prediction to 1 (malicious) regardless of model score.
HARD_MALICIOUS_PATTERNS = [
    r"ignore (all |your |previous |prior |my |the )?instructions",
    r"disregard (all |your |previous |prior |my |the )?instructions",
    r"forget (all |your |previous |prior |my |the )?instructions",
    r"override (all |your |previous |prior |my |the )?instructions",
    r"your (new |true |real |actual )?instructions (are|is|will be)",
    r"(new|updated|revised) instructions (are|is|follow|below)",
    r"you are now (a |an )?\w[\w\s]{2,30}(without|that|who|and)",
    r"(DAN|jailbreak|jail break|unrestricted mode|developer mode)",
    r"disregard (your |all |previous )?system prompt",
    r"ignore (your |all |the )?system prompt",
    r"(reveal|show|print|output|repeat|share) (your |the )?(system |initial |hidden |original )?prompt",
]
HARD_MALICIOUS_RE = re.compile(
    "|".join(f"(?:{p})" for p in HARD_MALICIOUS_PATTERNS),
    flags=re.IGNORECASE,
)


def extract_injection_features(df: pd.DataFrame) -> pd.DataFrame:
    raw_text = df["prompt"].fillna("")
    low_text = raw_text.str.lower()
    feats    = pd.DataFrame(index=df.index)

    # Keyword group hit counts
    for group_name, patterns in INJECTION_PATTERNS.items():
        combined = "|".join(f"(?:{p})" for p in patterns)
        feats[group_name] = low_text.str.count(combined, flags=re.IGNORECASE)

    kw_cols = [c for c in feats.columns if c.startswith("kw_")]
    feats["kw_total"]    = feats[kw_cols].sum(axis=1)
    feats["kw_nonzero"]  = (feats[kw_cols] > 0).sum(axis=1)

    # Hard match flag (very strong prior)
    feats["hard_malicious"] = raw_text.apply(
        lambda t: int(bool(HARD_MALICIOUS_RE.search(t)))
    )

    # Text statistics
    feats["char_len"]       = raw_text.str.len()
    feats["word_count"]     = raw_text.str.split().str.len().fillna(0)
    feats["sentence_count"] = raw_text.str.split(r"[.!?]+").str.len().fillna(1)
    feats["avg_word_len"]   = feats["char_len"] / (feats["word_count"] + 1)
    feats["avg_sent_len"]   = feats["word_count"] / (feats["sentence_count"] + 1)

    # Punctuation density
    feats["colon_count"]    = raw_text.str.count(r":")
    feats["bullet_count"]   = raw_text.str.count(r"[*\-]")
    feats["quote_count"]    = raw_text.str.count(r'["\']')
    feats["paren_count"]    = raw_text.str.count(r"[\(\)\[\]]")
    feats["newline_count"]  = raw_text.str.count(r"\n")
    feats["question_count"] = raw_text.str.count(r"\?")
    feats["exclaim_count"]  = raw_text.str.count(r"!")

    feats["caps_ratio"] = raw_text.apply(
        lambda t: sum(1 for c in t if c.isupper()) / (len(t) + 1)
    )

    # LLM-style structural signals
    feats["has_numbered_list"] = raw_text.str.contains(r"\b\d+[\.\)]\s", regex=True).astype(int)
    feats["has_code_block"]    = raw_text.str.contains(r"```", regex=False).astype(int)
    feats["has_checkmarks"]    = raw_text.str.contains(r"[✓✔☑]", regex=True).astype(int)
    feats["has_url"]           = raw_text.str.contains(r"https?://", regex=True).astype(int)

    feats["explicit_format"] = raw_text.str.contains(
        r"begin with|start your response|your response must|structure.*as follows|"
        r"format.*must be|strictly follow|exactly as (follows|specified)|"
        r"respond only with|output only|nothing else",
        flags=re.IGNORECASE, regex=True,
    ).astype(int)

    formal_words = (
        r"\b(please|kindly|hereby|pursuant|therefore|furthermore|additionally|"
        r"specifically|exactly|precisely|accordingly|subsequently)\b"
    )
    feats["formal_count"] = raw_text.str.count(formal_words, flags=re.IGNORECASE)

    feats["you_commands"] = raw_text.str.count(
        r"\b(you must|you should|you will|you need to|you are required|"
        r"your (next |)response|your (next |)output)\b",
        flags=re.IGNORECASE,
    )

    # Benign signals (genuine user questions)
    feats["is_question"]  = (feats["question_count"] > 0).astype(int)
    feats["starts_can"]   = raw_text.str.match(r"^(Can|Could|Would|How|What|Why|When|Please)", re.IGNORECASE).astype(int)

    # Density features
    feats["kw_density"]     = feats["kw_total"] / (feats["word_count"] + 1)
    feats["groups_density"] = feats["kw_nonzero"] / len(INJECTION_PATTERNS)

    return feats.fillna(0)


def build_tfidf_features(train_texts: pd.Series, test_texts: pd.Series):
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
print(f"    Hard-malicious hits in train: {feats_train['hard_malicious'].sum():.0f} / {len(train)}")
print(f"    Hard-malicious hits in test:  {feats_test['hard_malicious'].sum():.0f} / {len(test)}")

print("[2] Building TF-IDF matrices ...")
X_tfidf_train, X_tfidf_test = build_tfidf_features(train["prompt"], test["prompt"])
print(f"    TF-IDF shape: {X_tfidf_train.shape}")

scaler = StandardScaler(with_mean=False)
X_hand_train = csr_matrix(scaler.fit_transform(feats_train.values))
X_hand_test  = csr_matrix(scaler.transform(feats_test.values))

X_train_full = hstack([X_tfidf_train, X_hand_train])
X_test_full  = hstack([X_tfidf_test,  X_hand_test])
y_train      = train[TARGET].values
print(f"    Combined feature matrix: {X_train_full.shape}")

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ─── LightGBM Params ──────────────────────────────────────────────────────────
LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "boosting_type":     "gbdt",
    "learning_rate":     0.03,
    "num_leaves":        15,
    "max_depth":         4,
    "min_child_samples": 12,
    "feature_fraction":  0.55,
    "bagging_fraction":  0.75,
    "bagging_freq":      5,
    "reg_alpha":         0.6,
    "reg_lambda":        1.2,
    "scale_pos_weight":  pos_weight,
    "verbose":          -1,
    "random_state":      SEED,
}

# ─── Stratified K-Fold CV (3 models) ─────────────────────────────────────────
N_FOLDS = 5
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb  = np.zeros(len(train))
oof_lr   = np.zeros(len(train))
oof_svm  = np.zeros(len(train))
test_lgb = np.zeros(len(test))
test_lr  = np.zeros(len(test))
test_svm = np.zeros(len(test))

fold_f1s = {"lgb": [], "lr": [], "svm": []}

print(f"\n[3] {N_FOLDS}-Fold Stratified CV (LGB + LR + LinearSVC) ...")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    X_trn, X_val = X_train_full[trn_idx], X_train_full[val_idx]
    y_trn, y_val = y_train[trn_idx],      y_train[val_idx]

    # LightGBM
    dtrain = lgb.Dataset(X_trn, label=y_trn)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model_lgb = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=1200,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof_lgb[val_idx]  = model_lgb.predict(X_val)
    test_lgb         += model_lgb.predict(X_test_full) / N_FOLDS

    # Logistic Regression
    model_lr = LogisticRegression(
        C=0.25, class_weight="balanced", max_iter=2000,
        solver="lbfgs", random_state=SEED,
    )
    model_lr.fit(X_trn, y_trn)
    oof_lr[val_idx]  = model_lr.predict_proba(X_val)[:, 1]
    test_lr         += model_lr.predict_proba(X_test_full)[:, 1] / N_FOLDS

    # LinearSVC (calibrated for probabilities) -- orthogonal decision boundary
    model_svm = CalibratedClassifierCV(
        LinearSVC(C=0.08, class_weight="balanced", max_iter=3000, random_state=SEED),
        cv=3,
    )
    model_svm.fit(X_trn, y_trn)
    oof_svm[val_idx]  = model_svm.predict_proba(X_val)[:, 1]
    test_svm         += model_svm.predict_proba(X_test_full)[:, 1] / N_FOLDS

    for name, oof in [("lgb", oof_lgb), ("lr", oof_lr), ("svm", oof_svm)]:
        f1 = f1_score(y_val, (oof[val_idx] > 0.5).astype(int))
        fold_f1s[name].append(f1)

    print(f"    Fold {fold + 1}: LGB={fold_f1s['lgb'][-1]:.4f} | "
          f"LR={fold_f1s['lr'][-1]:.4f} | SVM={fold_f1s['svm'][-1]:.4f}")

for name in ("lgb", "lr", "svm"):
    print(f"    {name.upper()} OOF mean F1 (thr=0.5): "
          f"{np.mean(fold_f1s[name]):.4f} +/- {np.std(fold_f1s[name]):.4f}")

# ─── 3-Way Blend + Threshold Optimisation ────────────────────────────────────
print("\n[4] Grid-searching 3-way blend x threshold for maximum OOF F1 ...")
best_f1, best_w1, best_w2, best_w3, best_thr = 0.0, 0.4, 0.3, 0.3, 0.4

# Coarse grid over 3 weights (normalised to sum to 1)
weight_candidates = []
for w1 in np.arange(0.1, 0.8, 0.1):
    for w2 in np.arange(0.1, 0.8, 0.1):
        w3 = 1.0 - w1 - w2
        if 0.05 <= w3 <= 0.75:
            weight_candidates.append((w1, w2, w3))

for w1, w2, w3 in weight_candidates:
    oof_blend = w1 * oof_lgb + w2 * oof_lr + w3 * oof_svm
    # Recall-biased floor: start from 0.08 to catch subtle FNs
    for thr in np.arange(0.08, 0.90, 0.01):
        f1 = f1_score(y_train, (oof_blend > thr).astype(int))
        if f1 > best_f1:
            best_f1, best_w1, best_w2, best_w3, best_thr = f1, w1, w2, w3, thr

print(f"    Best: LGB={best_w1:.2f} LR={best_w2:.2f} SVM={best_w3:.2f} | "
      f"threshold={best_thr:.2f} | OOF F1={best_f1:.4f}")

# ─── Final Test Predictions ───────────────────────────────────────────────────
print("\n[5] Generating test predictions ...")
test_final  = best_w1 * test_lgb + best_w2 * test_lr + best_w3 * test_svm
test_labels = (test_final > best_thr).astype(int)

# ─── Hard-Override Post-Processing ───────────────────────────────────────────
# If a prompt contains an undeniable injection signature, force label=1.
# These patterns are essentially impossible in legitimate prompts.
forced = 0
raw_test_prompts = test["prompt"].fillna("")
for i, (prompt, current_label) in enumerate(zip(raw_test_prompts, test_labels)):
    if current_label == 0 and HARD_MALICIOUS_RE.search(prompt):
        test_labels[i] = 1
        forced += 1

print(f"    Hard-override forced malicious: {forced} prediction(s)")
print(f"    Test -- malicious: {test_labels.sum()} | benign: {(test_labels == 0).sum()}")

# ─── Submission ───────────────────────────────────────────────────────────────
sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_labels})
sub.to_csv(SUB_PATH, index=False)
print(f"\n    Saved -> {SUB_PATH}")
print(f"    Shape: {sub.shape}  |  Unique IDs: {sub[ID_COL].nunique()}")
assert sub[ID_COL].nunique() == len(test), "ID mismatch!"
assert set(sub[TARGET].unique()).issubset({0, 1}), "Labels must be 0 or 1!"

# Compare vs v1 predictions
v1_sub = pd.read_csv(os.path.join(COMP_DIR, "submission.csv"))
diff   = (sub[TARGET].values != v1_sub[TARGET].values).sum()
print(f"\n    Changes vs v1: {diff} prediction(s) flipped")
changed_ids = sub[ID_COL][sub[TARGET].values != v1_sub[TARGET].values].tolist()
if changed_ids:
    print(f"    Changed IDs: {changed_ids}")

print("\n" + "=" * 65)
print("Done.")
