import numpy as np
import pandas as pd

df = pd.read_csv('ia_P3_PrediccioResultats/portuguese_hs_students.csv')
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('G3', axis=1).values
y = (df_encoded['G3'].values >= 18).astype(int)

n = len(X)
split = int(n * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

k = 3
y_pred = []
for test_row in X_test:
    distances = [euclidean(test_row, train_row) for train_row in X_train]
    idx = np.argsort(distances)[:k]
    votes = y_train[idx]
    pred = int(np.mean(votes) >= 0.5)
    y_pred.append(pred)
y_pred = np.array(y_pred)

accuracy = np.mean(y_test == y_pred)
tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("Algoritmo utilizado: KNN (k=3)")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")