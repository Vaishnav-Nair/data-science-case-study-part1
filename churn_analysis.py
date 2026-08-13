import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)

# ---------- Load ----------
df = pd.read_csv('data/telco_churn.csv')
print("Shape:", df.shape)
print(df['Churn'].value_counts())

# ---------- Clean ----------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df.drop('customerID', axis=1, inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical features
cat_cols = df.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# ---------- kNN ----------
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train_s, y_train)
pred_knn = knn.predict(X_test_s)
proba_knn = knn.predict_proba(X_test_s)[:, 1]

results['kNN'] = {
    'accuracy': accuracy_score(y_test, pred_knn),
    'precision': precision_score(y_test, pred_knn),
    'recall': recall_score(y_test, pred_knn),
    'f1': f1_score(y_test, pred_knn),
    'roc_auc': roc_auc_score(y_test, proba_knn),
    'confusion_matrix': confusion_matrix(y_test, pred_knn).tolist()
}

# ---------- Neural Network (MLP) ----------
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42, early_stopping=True)
mlp.fit(X_train_s, y_train)
pred_mlp = mlp.predict(X_test_s)
proba_mlp = mlp.predict_proba(X_test_s)[:, 1]

results['NeuralNet'] = {
    'accuracy': accuracy_score(y_test, pred_mlp),
    'precision': precision_score(y_test, pred_mlp),
    'recall': recall_score(y_test, pred_mlp),
    'f1': f1_score(y_test, pred_mlp),
    'roc_auc': roc_auc_score(y_test, proba_mlp),
    'confusion_matrix': confusion_matrix(y_test, pred_mlp).tolist()
}

print("\n===== CHURN DATASET RESULTS =====")
for model, m in results.items():
    print(f"\n{model}:")
    for k, v in m.items():
        print(f"  {k}: {v}")

# Feature importance proxy via permutation for kNN, and via first-layer weights magnitude for MLP
from sklearn.inspection import permutation_importance
perm = permutation_importance(knn, X_test_s, y_test, n_repeats=10, random_state=42)
feat_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print("\nTop 8 features driving churn (kNN permutation importance):")
print(feat_imp.head(8))

import json
with open('churn_results.json', 'w') as f:
    json.dump(results, f, indent=2)

feat_imp.head(10).to_csv('churn_feature_importance.csv')
print("\nSaved results.")
