# Assignment 3: Bank Customer Churn Prediction using ANN
# Dataset: Churn_Modelling.csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("Churn_Modelling.csv")
print(df.head(20))

target = "Exited"
if target not in df.columns:
    raise ValueError("Expected target column 'Exited' in Churn_Modelling.csv")

# Remove identifier columns if present.
X = df.drop(columns=[target, "RowNumber", "CustomerId", "Surname"], errors="ignore")
y = df[target]

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Pipeline([
    ("prep", preprocessor),
    ("ann", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300,
                          early_stopping=True, random_state=42))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
