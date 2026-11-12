# Assignment 2: Email/SMS Spam Detection
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep the required columns
df = df[["v1", "v2"]].dropna()
df.columns = ["label", "message"]

# Convert text into numeric features
X = TfidfVectorizer(stop_words="english").fit_transform(df["message"])
y = df["label"].map({"ham": 0, "spam": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("\nKNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Classification Report:")
print(classification_report(y_test, knn_pred, target_names=["Ham", "Spam"]))

# SVM Model
svm = LinearSVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print("SVM Classification Report:")
print(classification_report(y_test, svm_pred, target_names=["Ham", "Spam"]))

# Test a sample message
message = ["Congratulations! You have won a free prize. Claim now!"]
sample = TfidfVectorizer(stop_words="english")
sample.fit(df["message"])
sample_vector = sample.transform(message)

prediction = svm.predict(sample_vector)
print("\nSample Message Prediction:",
      "Spam" if prediction[0] == 1 else "Ham")