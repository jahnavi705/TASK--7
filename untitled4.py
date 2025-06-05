import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

data = pd.read_csv('/content/breast-cancer.csv')

print("First 5 rows of the dataset:")
print(data.head())

print("\nColumn names before filtering:")
print(data.columns)

target_column_name = 'diagnosis' 
if target_column_name in data.columns:
    y = data[target_column_name]
    X = data.drop(target_column_name, axis=1)
else:
    print(f"Error: Target column '{target_column_name}' not found in the dataset.")
    exit()

X = X.select_dtypes(include=[np.number])

combined_data = pd.concat([X, y], axis=1)
combined_data = combined_data.dropna()

X = combined_data.drop(target_column_name, axis=1)
y = combined_data[target_column_name]

print(f"\nUnique values in the target column '{target_column_name}': {y.unique()}")

if y.dtype == 'object':
    print(f"Converting target column '{target_column_name}' to numeric...")
    mapping = {'M': 1, 'B': 0}
    y = y.map(mapping)
    y = y.dropna()
    X = X.loc[y.index]
    print(f"Conversion complete. Unique numeric values in target: {y.unique()}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train_scaled, y_train)
y_pred_linear = linear_svm.predict(X_test_scaled)

print("\n--- Linear Kernel SVM ---")
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=5)
grid.fit(X_train_scaled, y_train)
y_pred_rbf = grid.predict(X_test_scaled)

print("\n--- RBF Kernel SVM (Tuned) ---")
print(f"Best Parameters: {grid.best_params_}")
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

if X_train_scaled.shape[0] > 0:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    model_pca = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
    model_pca.fit(X_pca, y_train)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
    plt.title('SVM Decision Boundary with RBF Kernel (PCA projection)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Class')
    plt.show()
else:
    print("\nCannot plot decision boundary: X_train_scaled is empty after data cleaning.")