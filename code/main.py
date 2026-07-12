import os
import itertools
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    classification_report,
)

FEATURES_CSV = "features.csv"
MODEL_OUT = "classifier.joblib"

def load_synthetic(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.binomial(1, 0.3, n)

    EG = np.where(label == 0, rng.beta(8, 2, n), rng.beta(2, 5, n))
    RP = np.where(label == 0, rng.beta(7, 3, n), rng.beta(2, 6, n))


    SC = np.where(label == 0, rng.beta(6, 3, n), rng.beta(3, 6, n))
    return pd.DataFrame({
        "example_id": range(n),
        "doc_id": range(n),  
        "EG": EG, "RP": RP, "SC": SC,
        "label": label,
    })


if os.path.exists(FEATURES_CSV):
    df = pd.read_csv(FEATURES_CSV)
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    df["label"] = df["label"].astype(int)
else:
    df = load_synthetic()

if "doc_id" not in df.columns:
    df["doc_id"] = df["example_id"]

def group_split(df, groups, test_size, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_a, idx_b = next(gss.split(df, groups=groups))
    return df.iloc[idx_a].reset_index(drop=True), df.iloc[idx_b].reset_index(drop=True)

train, temp = group_split(df, df["doc_id"], test_size=0.4)
val, test = group_split(temp, temp["doc_id"], test_size=0.5)

FEATURES = ["EG", "RP", "SC"]
X_train, y_train = train[FEATURES], train["label"]
X_val, y_val = val[FEATURES], val["label"]


X_test, y_test = test[FEATURES], test["label"]

best_auc_ws, best_weights = 0.0, None
for w1, w2 in itertools.product(np.arange(0, 1.05, 0.1), repeat=2):
    w3 = 1 - w1 - w2
    if w3 < -1e-9 or w3 > 1 + 1e-9:
        continue
    U_val = w1 * val["EG"] + w2 * val["RP"] + w3 * val["SC"]
    auc = roc_auc_score(y_val, 1 - U_val)
    if auc > best_auc_ws:
        best_auc_ws, best_weights = auc, (round(w1, 2), round(w2, 2), round(w3, 2))

clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_train, y_train)
proba_val_lr = clf_lr.predict_proba(X_val)[:, 1]
auc_lr = roc_auc_score(y_val, proba_val_lr)
print(f"AUC: {auc_lr:.3f}")




clf_gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_gb.fit(X_train, y_train)
proba_val_gb = clf_gb.predict_proba(X_val)[:, 1]
auc_gb = roc_auc_score(y_val, proba_val_gb)
print(f"AUC: {auc_gb:.3f}")

def report(name, scores):
    auroc = roc_auc_score(y_val, scores)
    auprc = average_precision_score(y_val, scores)
    print(f"{name:25s} {auroc:7.3f} {auprc:8.3f}")
    return auroc

for metric in FEATURES:
    report(f"only_{metric}", 1 - val[metric])

w1, w2, w3 = best_weights
report("weighted_sum", 1 - (w1 * val["EG"] + w2 * val["RP"] + w3 * val["SC"]))

report("logistic_regression", proba_val_lr)
report("gradient_boosting", proba_val_gb)

if auc_gb > auc_lr:
    best_clf, best_proba_val, best_name = clf_gb, proba_val_gb, "gradient_boosting"
else:
    best_clf, best_proba_val, best_name = clf_lr, proba_val_lr, "logistic_regression"

fpr, tpr, thresholds = roc_curve(y_val, best_proba_val)
best_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[best_idx]


proba_test = best_clf.predict_proba(X_test)[:, 1]
test_auroc = roc_auc_score(y_test, proba_test)
test_auprc = average_precision_score(y_test, proba_test)
pred_test = (proba_test >= best_threshold).astype(int)

print(f"Test AUROC: {test_auroc:.3f}")
print(f"Test AU-PRC: {test_auprc:.3f}\n")
print(classification_report(
    y_test, pred_test,
    target_names=["not_hallucination", "hallucination"],
    zero_division=0,
))


joblib.dump(
    {"model": best_clf, "threshold": float(best_threshold), "features": FEATURES},
    MODEL_OUT,
)
