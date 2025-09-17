import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/car_price_model.pkl")

st.title("🚗 Car Price Prediction App")

st.write("Fill in the details below to estimate the car's selling price.")

# Input fields
year = st.number_input("Year of Manufacture", 1990, 2025, 2015)
km_driven = st.number_input("Kilometers Driven", 0, 500000, 20000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Number of Previous Owners", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

mileage = st.text_input("Mileage (e.g. '21.5 kmpl' or '18.0 km/kg')", "21.5 kmpl")
engine = st.text_input("Engine Capacity (e.g. '1197 CC')", "1197 CC")
max_power = st.text_input("Max Power (e.g. '82 bhp')", "82 bhp")
seats = st.number_input("Number of Seats", 2, 10, 5)

if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }])

    # --- Preprocessing ---
    # Clean numeric columns
    input_data['mileage'] = input_data['mileage'].str.replace(' kmpl','').str.replace(' km/kg','').astype(float)
    input_data['engine'] = input_data['engine'].str.replace(' CC','').astype(float)
    input_data['max_power'] = input_data['max_power'].str.replace(' bhp','').astype(float)

    # Match training dummy variables
    input_data = pd.get_dummies(input_data)

    # Align columns with training set
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Selling Price: ₹ {prediction:,.2f}")
