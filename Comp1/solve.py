import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv(r'C:\Users\jaish\Downloads\train.csv')
test = pd.read_csv(r'C:\Users\jaish\Downloads\test.csv')

print(f"Train: {len(train)} rows")
print(f"Test: {len(test)} rows")
print("\nClass distribution:")
print(train['risk_category'].value_counts())

X_train = train['sentence'].fillna('').values
y_train = train['risk_category'].values
X_test = test['sentence'].fillna('').values

word_tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=100000,
    sublinear_tf=True,
    min_df=1,
    analyzer='word',
    strip_accents='unicode',
)

char_tfidf = TfidfVectorizer(
    ngram_range=(3, 5),
    max_features=50000,
    sublinear_tf=True,
    min_df=2,
    analyzer='char_wb',
    strip_accents='unicode',
)

print("\nFitting TF-IDF features...")
X_train_word = word_tfidf.fit_transform(X_train)
X_test_word = word_tfidf.transform(X_test)

X_train_char = char_tfidf.fit_transform(X_train)
X_test_char = char_tfidf.transform(X_test)

X_train_feat = hstack([X_train_word, X_train_char])
X_test_feat = hstack([X_test_word, X_test_char])

print(f"Feature matrix: {X_train_feat.shape}")

clf = LinearSVC(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)

print("\nCross-validating (5-fold macro F1)...")
cv_scores = cross_val_score(clf, X_train_feat, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
print(f"CV Macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\nFitting on all training data...")
clf.fit(X_train_feat, y_train)

predictions = clf.predict(X_test_feat)

print("\nPrediction distribution:")
print(pd.Series(predictions).value_counts())

submission = pd.DataFrame({'id': test['id'], 'risk_category': predictions})
out_path = r'C:\Users\jaish\Downloads\submission.csv'
submission.to_csv(out_path, index=False)
print(f"\nSaved {len(submission)} rows to {out_path}")
print(submission.head())
