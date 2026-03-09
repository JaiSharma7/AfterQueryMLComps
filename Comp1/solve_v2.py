
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv(r"C:/Users/jaish/Downloads/train.csv")
test = pd.read_csv(r"C:/Users/jaish/Downloads/test.csv")
X_train = train["sentence"].fillna("").values
y_train = train["risk_category"].values
X_test = test["sentence"].fillna("").values

CATEGORY_KEYWORDS = {
    "competitive_risk": [
        "competition", "competitive", "competitor", "market share", "pricing pressure",
        "new entrants", "consumer preferences", "substitute", "differentiate",
        "competitive advantage", "market position", "rivals", "competing products",
        "competing services", "product competition", "price war"
    ],
    "cyber_technology_risk": [
        "cybersecurity", "cyber", "data breach", "hack", "malware", "ransomware", "phishing",
        "IT failure", "technology failure", "data security", "information security",
        "network security", "data privacy", "unauthorized access", "cyber attack",
        "cyber incident", "systems failure", "technology infrastructure", "encryption",
        "vulnerability", "penetration testing", "incident response", "data protection"
    ],
    "financial_credit_risk": [
        "credit risk", "credit rating", "debt", "liquidity", "cash flow", "impairment",
        "goodwill impairment", "interest expense", "borrowing", "leverage",
        "financial obligations", "covenant", "default", "credit facility",
        "revolving credit", "debt financing", "credit loss", "allowance", "write-off",
        "write-down", "fair value", "discounted cash flow", "refinancing", "solvency"
    ],
    "litigation_risk": [
        "lawsuit", "litigation", "legal proceedings", "legal action", "class action",
        "settlement", "damages", "plaintiff", "defendant", "court", "judicial",
        "arbitration", "patent", "intellectual property", "infringement",
        "trade secret", "legal claim", "legal challenge", "government investigation",
        "enforcement action", "indemnification", "legal liability", "judgment"
    ],
    "market_risk": [
        "interest rate", "foreign exchange", "currency", "commodity price", "oil price",
        "inflation", "macroeconomic", "economic conditions", "recession", "market volatility",
        "stock price", "equity price", "market downturn", "economic slowdown",
        "hedging", "derivative", "exchange rate", "monetary policy", "federal reserve",
        "central bank", "yield"
    ],
    "operational_risk": [
        "supply chain", "manufacturing", "workforce", "employees", "natural disaster",
        "third-party vendor", "supplier", "production", "distribution", "logistics",
        "quality control", "product recall", "business continuity", "pandemic",
        "labor", "union", "staffing", "outsourcing", "vendor", "facilities",
        "capacity", "raw materials", "third party"
    ],
    "regulatory_risk": [
        "regulation", "regulatory", "compliance", "government", "legislation", "law",
        "policy", "tax", "tariff", "trade restriction", "FDA", "SEC", "FTC",
        "antitrust", "GDPR", "privacy law", "environmental regulation", "permit",
        "license", "approval", "enforcement", "penalty", "fine", "sanction",
        "congressional", "legislative", "statutory", "regulator"
    ],
    "strategic_risk": [
        "acquisition", "merger", "joint venture", "strategic alliance",
        "partnership", "international expansion", "restructuring", "divestiture",
        "spin-off", "integration", "geopolitical", "political risk",
        "strategic initiative", "transformation", "diversification", "new market",
        "emerging market", "growth strategy", "business combination", "takeover",
        "synergy", "strategic transaction"
    ],
    "other_risk": [
        "forward-looking", "forward looking", "estimate", "assumption",
        "reputational", "reputation", "brand", "sustainability", "ESG",
        "climate change", "public health", "demographic"
    ]
}

def make_keyword_features(texts):
    n = len(texts)
    categories = list(CATEGORY_KEYWORDS.keys())
    n_cats = len(categories)
    features = np.zeros((n, n_cats))
    for i, text in enumerate(texts):
        text_lower = text.lower()
        for j, (cat, keywords) in enumerate(CATEGORY_KEYWORDS.items()):
            count = sum(1 for kw in keywords if kw.lower() in text_lower)
            features[i, j] = count
    return csr_matrix(features)

print("Building TF-IDF features...")
word_tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=100000, sublinear_tf=True,
    min_df=1, analyzer="word", strip_accents="unicode")
char_tfidf = TfidfVectorizer(ngram_range=(3, 5), max_features=50000, sublinear_tf=True,
    min_df=2, analyzer="char_wb", strip_accents="unicode")

X_train_word = word_tfidf.fit_transform(X_train)
X_test_word = word_tfidf.transform(X_test)
X_train_char = char_tfidf.fit_transform(X_train)
X_test_char = char_tfidf.transform(X_test)

print("Building keyword features...")
X_train_kw = make_keyword_features(X_train)
X_test_kw = make_keyword_features(X_test)

X_train_feat = hstack([X_train_word, X_train_char, X_train_kw * 5])
X_test_feat = hstack([X_test_word, X_test_char, X_test_kw * 5])
print(f"Feature matrix: {X_train_feat.shape}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Tuning LinearSVC C...")
best_score = 0
best_C = 0.5
for C in [0.1, 0.3, 0.5, 1.0, 2.0]:
    clf = LinearSVC(C=C, class_weight="balanced", max_iter=2000, random_state=42)
    scores = cross_val_score(clf, X_train_feat, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"  C={C}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_C = C

print(f"Best C={best_C}, CV Macro F1={best_score:.4f}")
final_clf = LinearSVC(C=best_C, class_weight="balanced", max_iter=2000, random_state=42)
final_clf.fit(X_train_feat, y_train)
predictions = final_clf.predict(X_test_feat)

print("Prediction distribution:")
print(pd.Series(predictions).value_counts())

submission = pd.DataFrame({"id": test["id"], "risk_category": predictions})
out_path = r"C:/Users/jaish/Downloads/submission.csv"
submission.to_csv(out_path, index=False)
print(f"Saved {len(submission)} rows to {out_path}")
