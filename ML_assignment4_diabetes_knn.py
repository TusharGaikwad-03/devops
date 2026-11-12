# Assignment 4: Diabetes Prediction using KNN
# Dataset: diabetes.csv; target column is commonly named Outcome.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,precision_score, recall_score)

df = pd.read_csv("diabetes.csv")
print(df.head())
target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

# Some diabetes datasets encode impossible zero measurements as missing.
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    if col in X.columns:
        X[col] = X[col].replace(0, float("nan"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

cm = confusion_matrix(y_test, pred)
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
error_rate = 1 - accuracy

print("Confusion Matrix:\n", cm)
print("Accuracy:", round(accuracy, 4))
print("Error Rate:", round(error_rate, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
