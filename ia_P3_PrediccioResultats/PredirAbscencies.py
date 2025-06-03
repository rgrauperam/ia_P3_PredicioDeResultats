import numpy as np
import pandas as pd

df = pd.read_csv('ia_P3_PrediccioResultats\portuguese_hs_students.csv')

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('absences', axis=1).values
y = df_encoded['absences'].values

n = len(X)
split = int(n * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

best_feature = None
best_threshold = None
best_mse = float('inf')
best_left_value = None
best_right_value = None

for feature in range(X_train.shape[1]):
    thresholds = np.unique(X_train[:, feature])
    for threshold in thresholds:
        left_mask = X_train[:, feature] <= threshold
        right_mask = X_train[:, feature] > threshold
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            continue
        left_mean = np.mean(y_train[left_mask])
        right_mean = np.mean(y_train[right_mask])
        left_mse = np.mean((y_train[left_mask] - left_mean) ** 2)
        right_mse = np.mean((y_train[right_mask] - right_mean) ** 2)
        mse = (np.sum(left_mask) * left_mse + np.sum(right_mask) * right_mse) / len(y_train)
        if mse < best_mse:
            best_mse = mse
            best_feature = feature
            best_threshold = threshold
            best_left_value = left_mean
            best_right_value = right_mean

y_pred = []
for row in X_test:
    if row[best_feature] <= best_threshold:
        y_pred.append(best_left_value)
    else:
        y_pred.append(best_right_value)
y_pred = np.array(y_pred)

mse = np.mean((y_test - y_pred) ** 2)
rmse = mse ** 0.5
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
ss_res = np.sum((y_test - y_pred) ** 2)
r2 = 1 - ss_res / ss_tot

print("Algorisme utilitzat: Àrbre de Decisió de Regressió (profunditat 1)")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
