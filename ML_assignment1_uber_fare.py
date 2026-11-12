import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Preprocessing
df = pd.read_csv("uber.csv")
df = df.dropna()
df = df[(df.fare_amount > 0) & (df.passenger_count > 0)]

# 2. Haversine Distance
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

df["distance_km"] = haversine(
    df.pickup_latitude, df.pickup_longitude,
    df.dropoff_latitude, df.dropoff_longitude)

df = df[df.distance_km.between(0.1, 200)]

# 3. Box Plot and Outliers
sns.boxplot(x=df["fare_amount"])
plt.title("Uber Fare Outliers")
plt.show()

q1, q3 = df.fare_amount.quantile([0.25, 0.75])
iqr = q3 - q1
df = df[df.fare_amount.between(q1-1.5*iqr, q3+1.5*iqr)]

# 4. Correlation Heatmap
sns.heatmap(df[["fare_amount", "passenger_count",
                "distance_km"]].corr(), annot=True)
plt.show()

# 5. Train-Test Split
X = df[["distance_km", "passenger_count"]]
y = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 6. Regression Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n", name)
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2 Score:", r2_score(y_test, pred))

# 7. Model Comparison using Matplotlib
plt.bar(models.keys(), [
    mean_squared_error(y_test, models[name].predict(X_test))
    for name in models
])
plt.title("Model MSE Comparison")
plt.show()
