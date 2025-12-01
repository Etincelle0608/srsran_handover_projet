!pip install openpyxl matplotlib scikit-learn pandas seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from google.colab import files
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform

pd.set_option("display.max_columns", None)
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_excel(filename)

print("Dimensions :", df.shape)
df.head()
target = "pci"

# pci doit être binaire : 7 ou 8 → conversion en 0/1
df[target] = df[target].replace({7: 0, 8: 1}).astype(int)

y = df[target]
X = df.drop(columns=[target])

print("Problème : CLASSIFICATION BINAIRE (handover success/fail)")
print("Distribution :", y.value_counts())
numeric_cols = X.select_dtypes(include=["int", "float"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Train :", X_train.shape, " Test :", X_test.shape)
rf = RandomForestClassifier(random_state=42)

param_dist = {
    "rf__n_estimators": randint(200, 1500),
    "rf__max_depth": randint(5, 40),
    "rf__min_samples_split": randint(2, 20),
    "rf__min_samples_leaf": randint(1, 10),
    "rf__max_features": ["sqrt"],
    "rf__bootstrap": [True, False]
}

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("rf", rf)
])

start_time = time.time()

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=40,
    scoring="accuracy",
    random_state=42,
    cv=3,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train)

train_total_time = time.time() - start_time
best_model = search.best_estimator_

start_pred = time.time()
y_test_pred = best_model.predict(X_test)
pred_test_time = time.time() - start_pred

start_pred2 = time.time()
y_train_pred = best_model.predict(X_train)
pred_train_time = time.time() - start_pred2

y_train_proba = best_model.predict_proba(X_train)[:, 1]
y_test_proba  = best_model.predict_proba(X_test)[:, 1]

print("Temps d'entraînement (s) :", train_total_time)
print("Temps prédiction Train (s):", pred_train_time)
print("Temps prédiction Test (s):", pred_test_time)

metrics = {
    "Train Accuracy": accuracy_score(y_train, y_train_pred),
    "Test Accuracy": accuracy_score(y_test, y_test_pred),
    "Train Precision": precision_score(y_train, y_train_pred),
    "Test Precision": precision_score(y_test, y_test_pred),
    "Train Recall": recall_score(y_train, y_train_pred),
    "Test Recall": recall_score(y_test, y_test_pred),
    "Train AUC": roc_auc_score(y_train, y_train_proba),
    "Test AUC": roc_auc_score(y_test, y_test_proba),
    "Training Time (s)": train_total_time,
    "Inference Time Train (s)": pred_train_time,
    "Inference Time Test (s)": pred_test_time
}

pd.DataFrame.from_dict(metrics, orient="index")

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de Confusion - Test Set")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.show()
fpr, tpr, th = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {metrics['Test AUC']:.3f}")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Courbe ROC")
plt.legend()
plt.grid(True)
plt.show()

# reconstruction des noms de features après OHE
encoded_features = []
for name, trans, cols in best_model.named_steps["preprocess"].transformers_:
    if name == "num":
        encoded_features.extend(cols)
    elif name == "cat":
        encoded_features.extend(
            trans["encoder"].get_feature_names_out(cols)
        )

importances = best_model.named_steps["rf"].feature_importances_
feat_imp = pd.DataFrame({
    "feature": encoded_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top_feature = feat_imp.iloc[0]
print(" Le paramètre le plus important pour déterminer le handover est :")
print(f"   {top_feature['feature']} (importance = {top_feature['importance']:.3f})")

# Affichage du Top 10
top10=feat_imp.head(10)
top10
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=top10, palette="viridis", hue="feature", legend=False)
plt.title("Top 9 des Features les Plus Importants")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
