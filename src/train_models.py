import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load cleaned data
data_path = os.path.join("data", "cleaned_car_data.csv")
df = pd.read_csv(data_path)

# Encode categorical columns
le = LabelEncoder()
for col in ['fuel', 'seller_type', 'transmission', 'owner']:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Save model
os.makedirs("../models", exist_ok=True)
with open("../models/car_price_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("[INFO] Model saved at models/car_price_model.pkl")
