
"""Train and evaluate TrustLens using the official fixed train/test split."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
import joblib

BASE = "https://raw.githubusercontent.com/trannguyenthaituan251209/vietnamese_sms_dataset/main/"
train = pd.read_csv(BASE + "train.csv").dropna(subset=["message", "label"])
test = pd.read_csv(BASE + "test.csv").dropna(subset=["message", "label"])
for df in (train, test):
    df["message"] = df["message"].astype(str)
    df["label"] = pd.to_numeric(df["label"]).astype(int)

X_train, y_train = train.message, train.label
X_test, y_test = test.message, test.label

baseline = DummyClassifier(strategy="most_frequent").fit(X_train.to_frame(), y_train)
baseline_pred = baseline.predict(X_test.to_frame())

features = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             max_features=90000, sublinear_tf=True))
])
model = Pipeline([
    ("features", features),
    ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("TEST SIZE:", len(test))
print("CLASS DISTRIBUTION:", y_test.value_counts().to_dict())
print("\nBASELINE (majority class)")
print(classification_report(y_test, baseline_pred, zero_division=0))
print("confusion matrix:\n", confusion_matrix(y_test, baseline_pred))
print("\nTRUSTLENS MODEL")
print(classification_report(y_test, pred, zero_division=0))
print("confusion matrix:\n", confusion_matrix(y_test, pred))
print("Scam precision:", precision_score(y_test, pred, pos_label=1, zero_division=0))
print("Scam recall:", recall_score(y_test, pred, pos_label=1, zero_division=0))
print("Scam F1:", f1_score(y_test, pred, pos_label=1, zero_division=0))

joblib.dump(model, "trustlens_model.joblib")
print("\nSaved trustlens_model.joblib")
