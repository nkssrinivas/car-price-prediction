import pickle
import pandas as pd

# Load trained model
with open("models/car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input (values must match encoded categories)
sample_input = {
    'year': 2018,
    'km_driven': 25000,
    'fuel': 2,           # Petrol/Diesel/CNG encoded
    'seller_type': 1,    # Dealer/Individual encoded
    'transmission': 1,   # Manual/Automatic encoded
    'owner': 0,          # First owner / etc.
    'mileage': 20.4,
    'engine': 1197,
    'max_power': 82,
    'seats': 5
}

df = pd.DataFrame([sample_input])
prediction = model.predict(df)

print(f"Predicted Selling Price: {prediction[0]:.2f} INR")
