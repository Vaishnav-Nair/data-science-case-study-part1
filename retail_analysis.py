import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

# ---------- Load ----------
df = pd.read_csv('data/online_retail.csv', encoding='utf-8-sig')
df.columns = df.columns.str.strip()  # remove hidden whitespace/BOM from column names
print("Columns:", df.columns.tolist())
print("Raw shape:", df.shape)

# ---------- Clean ----------
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='mixed', dayfirst=True)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print("Cleaned shape:", df.shape)

# ---------- RFM feature engineering per customer ----------
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum'),
    AvgBasketValue=('TotalPrice', 'mean'),
    UniqueProducts=('StockCode', 'nunique')
).reset_index()

print("RFM table shape:", rfm.shape)
print(rfm.describe())

# ---------- Engineer binary target: "High-value customer" = top 25% by Monetary ----------
threshold = rfm['Monetary'].quantile(0.75)
rfm['HighValue'] = (rfm['Monetary'] >= threshold).astype(int)
print("\nHighValue distribution:")
print(rfm['HighValue'].value_counts())

# Drop CustomerID and Monetary (to avoid leakage since target derived from it) — keep other RFM features
X = rfm[['Recency', 'Frequency', 'AvgBasketValue', 'UniqueProducts']]
y = rfm['HighValue']

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

# ---------- Neural Network ----------
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

print("\n===== RETAIL (HIGH-VALUE CUSTOMER) RESULTS =====")
for model, m in results.items():
    print(f"\n{model}:")
    for k, v in m.items():
        print(f"  {k}: {v}")

from sklearn.inspection import permutation_importance
perm = permutation_importance(knn, X_test_s, y_test, n_repeats=10, random_state=42)
feat_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print("\nFeature importance (kNN permutation):")
print(feat_imp)

import json
with open('retail_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved results.")
