import numpy as np
import pandas as pd

df = pd.read_csv('ia_P3_PrediccioResultats\portuguese_hs_students.csv')

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('G3', axis=1).values
y = (df_encoded['G3'].values >= 10).astype(int)

n = len(X)
split = int(n * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

best_feature = None
best_threshold = None
best_gini = float('inf')
best_left_class = None
best_right_class = None

for feature in range(X_train.shape[1]):
    thresholds = np.unique(X_train[:, feature])
    for threshold in thresholds:
        left_mask = X_train[:, feature] <= threshold
        right_mask = X_train[:, feature] > threshold
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            continue
        left_class = int(np.mean(y_train[left_mask]) >= 0.5)
        right_class = int(np.mean(y_train[right_mask]) >= 0.5)
        
        left_gini = 1 - np.sum([(np.mean(y_train[left_mask] == c))**2 for c in [0,1]])
        right_gini = 1 - np.sum([(np.mean(y_train[right_mask] == c))**2 for c in [0,1]])
        gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / len(y_train)
        if gini < best_gini:
            best_gini = gini
            best_feature = feature
            best_threshold = threshold
            best_left_class = left_class
            best_right_class = right_class

y_pred = []
for row in X_test:
    if row[best_feature] <= best_threshold:
        y_pred.append(best_left_class)
    else:
        y_pred.append(best_right_class)
y_pred = np.array(y_pred)

accuracy = np.mean(y_test == y_pred)
tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("Algorisme utilitzat: Àrbre de Decisió de Classificació (profunditat 1)")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
