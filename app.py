import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from database import create_table, insert_prediction, fetch_predictions

# Initialize database (uncomment if using)
create_table()

# Load model
model = joblib.load("random_forest_model.pkl")

# Load feature columns
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Load training bin edges for price_original
with open("price_bins.json", "r") as f:
    training_bin_edges = json.load(f)

st.set_page_config(page_title="Game Success Predictor", layout="wide")

st.title("Game Success Prediction Dashboard")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Make Prediction", "Prediction History"]
)

# ===============================
# PREDICTION PAGE
# ===============================

if menu == "Make Prediction":

    st.subheader("Enter Game Information")

    col1, col2 = st.columns(2)

    with col1:
        game_name = st.text_input("Game Name")
        price_original_input = st.number_input("Original Price ($)", min_value=0.0)
        discount_input = st.number_input("Discount (%)", min_value=0.0)

    with col2:
        win_ui = st.selectbox("Windows Support", ["Yes", "No"])
        mac_ui = st.selectbox("Mac Support", ["Yes", "No"])
        linux_ui = st.selectbox("Linux Support", ["Yes", "No"])

        # Convert to numeric for model
        win = 1 if win_ui == "Yes" else 0
        mac = 1 if mac_ui == "Yes" else 0
        linux = 1 if linux_ui == "Yes" else 0

    if st.button("Predict Game Success"):
        if game_name == "":
            st.warning("Game Name cannot be empty")
        elif win + mac + linux == 0:
            st.warning("Please select at least one supported platform (Windows, Mac, or Linux).")
        else:
            # Apply log transformation (same as training)
            price_original_log = np.log1p(price_original_input)
            discount_log = np.log1p(discount_input)

            # Create dictionary with all expected columns
            input_dict = {col: 0 for col in feature_columns}

            # Fill base features
            if "price_original" in input_dict:
                input_dict["price_original"] = price_original_log

            if "discount" in input_dict:
                input_dict["discount"] = discount_log

            if "win" in input_dict:
                input_dict["win"] = win

            if "mac" in input_dict:
                input_dict["mac"] = mac

            if "linux" in input_dict:
                input_dict["linux"] = linux

            # Recreate interaction feature
            if "mac_and_linux" in input_dict:
                input_dict["mac_and_linux"] = mac * linux

            # Handle price bins (CORRECTED LOGIC)
            # Initialize one-hot encoded columns to 0
            input_dict["price_original_medium-low"] = 0
            input_dict["price_original_medium-high"] = 0
            input_dict["price_original_high"] = 0

            # Assign based on the calculated bins for the log-transformed price
            # Note: training_bin_edges[0] is the min, training_bin_edges[4] is the max.
            # The bins are (lower_bound, upper_bound].
            if training_bin_edges[1] < price_original_log <= training_bin_edges[2]:
                input_dict["price_original_medium-low"] = 1
            elif training_bin_edges[2] < price_original_log <= training_bin_edges[3]:
                input_dict["price_original_medium-high"] = 1
            elif training_bin_edges[3] < price_original_log <= training_bin_edges[4]:
                input_dict["price_original_high"] = 1
            # If price_original_log is <= training_bin_edges[1], it's 'low', and all bin columns remain 0.

            # Convert to DataFrame
            input_data = pd.DataFrame([input_dict])

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]


            insert_prediction(
                game_name,
                price_original_input,
                discount_input,
                win,
                mac,
                linux,
                int(prediction),
                float(probability)
            )

            if prediction == 1:
                st.success(f"High Potential Game! (Confidence: {probability:.2f})")
            else:
                st.error(f"Standard/Low Potential Game (Confidence: {probability:.2f})")

# ===============================
# HISTORY PAGE
# ===============================

elif menu == "Prediction History":

    st.subheader("Prediction History")

    data = fetch_predictions()

    if data:
        df = pd.DataFrame(data, columns=[
            "ID",
            "Timestamp",
            "Game Name",
            "Original Price",
            "Discount",
            "Windows",
            "Mac",
            "Linux",
            "Prediction",
            "Confidence"
        ])

        total_predictions = len(df)
        high_potential = len(df[df["Prediction"] == 1])
        success_rate = (high_potential / total_predictions) * 100 if total_predictions > 0 else 0

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", total_predictions)
        col2.metric("High Potential Games", high_potential)
        col3.metric("Success Rate (%)", f"{success_rate:.2f}")

        df["Prediction"] = df["Prediction"].map({
            0: "Low Potential",
            1: "High Potential"
        })

        st.dataframe(df, use_container_width=True)

    else:
        st.info("No predictions recorded yet.")