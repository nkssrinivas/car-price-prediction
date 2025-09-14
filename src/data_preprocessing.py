import pandas as pd
import numpy as np
import os

# Load dataset
data_path = os.path.join("data", "Car details v3.csv")
df = pd.read_csv(data_path)

# Drop duplicates & missing values
df = df.drop_duplicates()
df = df.dropna()

# Function to extract numeric values (for mileage, engine, power)
def extract_numeric(x):
    try:
        return float(str(x).split()[0])
    except:
        return np.nan

df['mileage'] = df['mileage'].apply(extract_numeric)
df['engine'] = df['engine'].apply(extract_numeric)
df['max_power'] = df['max_power'].apply(extract_numeric)

# Drop columns not useful
df = df.drop(['torque', 'name'], axis=1)

# Save cleaned dataset
os.makedirs("data", exist_ok=True)
cleaned_path = os.path.join("data", "cleaned_car_data.csv")
df.to_csv(cleaned_path, index=False)

print(f"[INFO] Cleaned dataset saved at {cleaned_path}")
