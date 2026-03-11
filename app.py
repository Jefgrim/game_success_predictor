import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# MUST be the first Streamlit command
st.set_page_config(page_title="Game Success Predictor", layout="wide")

from database import create_table, insert_prediction, fetch_predictions

# Initialize database
create_table()


# Load model and assets
@st.cache_resource
def load_model_and_assets():
    model = joblib.load("random_forest_model.pkl")
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    with open("price_bins.json", "r") as f:
        training_bin_edges = json.load(f)
    return model, feature_columns, training_bin_edges


model, feature_columns, training_bin_edges = load_model_and_assets()

st.title("Game Success Prediction Dashboard")

menu = st.sidebar.selectbox("Navigation", ["Make Prediction", "Prediction History"])

# ===============================
# PREDICTION PAGE
# ===============================
if menu == "Make Prediction":
    st.subheader("Enter Game Information")

    col1, col2 = st.columns(2)

    with col1:
        game_name = st.text_input("Game Name")
        price_original_input = st.number_input("Original Price ($)", min_value=0.0, value=19.99)
        discount_input = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0)

    with col2:
        win_ui = st.selectbox("Windows Support", ["Yes", "No"])
        mac_ui = st.selectbox("Mac Support", ["Yes", "No"])
        linux_ui = st.selectbox("Linux Support", ["Yes", "No"])

        # Convert to numeric for model
        win = 1 if win_ui == "Yes" else 0
        mac = 1 if mac_ui == "Yes" else 0
        linux = 1 if linux_ui == "Yes" else 0

    if st.button("Predict Game Success", type="primary"):
        if game_name == "":
            st.warning("Game Name cannot be empty")
        elif win + mac + linux == 0:
            st.warning("Please select at least one supported platform (Windows, Mac, or Linux).")
        else:
            # Apply log transformation (same as training)
            price_original_log = np.log1p(price_original_input)
            discount_log = np.log1p(discount_input)

            # Create dictionary with all expected columns initialized to 0
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

            # Handle price bins dynamically based on training quartiles
            if price_original_log <= training_bin_edges[1]:
                if "price_original_low" in input_dict: input_dict["price_original_low"] = 1
            elif price_original_log <= training_bin_edges[2]:
                if "price_original_medium-low" in input_dict: input_dict["price_original_medium-low"] = 1
            elif price_original_log <= training_bin_edges[3]:
                if "price_original_medium-high" in input_dict: input_dict["price_original_medium-high"] = 1
            else:
                if "price_original_high" in input_dict: input_dict["price_original_high"] = 1

            # Convert to DataFrame and enforce strict column order
            input_data = pd.DataFrame([input_dict])
            input_data = input_data[feature_columns]

            # Predict
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            # Insert to DB (Uncommented and active)
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

            st.markdown("---")
            st.subheader("Prediction Results")

            # Result Layout
            res_col1, res_col2 = st.columns([1, 1.5])

            with res_col1:
                if prediction == 1:
                    st.success(f"**{game_name}** has HIGH potential!")
                else:
                    st.error(f"**{game_name}** has STANDARD/LOW potential.")

                # Probability Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={'text': "Probability of Success"},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightcoral"},
                            {'range': [50, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.write("### Why did the model make this decision?")
                st.write(
                    "This chart shows the **Feature Importances** of the Random Forest model. It highlights which factors weighed the heaviest in calculating the success probability.")

                # Feature Importance Bar Chart
                importances = model.feature_importances_
                feat_df = pd.DataFrame({
                    "Feature": feature_columns,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=True)

                # Make feature names more readable
                feat_df["Feature"] = feat_df["Feature"].str.replace("_", " ").str.title()

                fig_bar = px.bar(
                    feat_df,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    color="Importance",
                    color_continuous_scale="Viridis",
                    title="Model Decision Drivers"
                )
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

# ===============================
# HISTORY PAGE
# ===============================
elif menu == "Prediction History":
    st.subheader("Prediction History")

    # Fetch from DB (Uncommented and active)
    data = fetch_predictions()

    if data:
        df = pd.DataFrame(data, columns=[
            "ID", "Timestamp", "Game Name", "Original Price", "Discount",
            "Windows", "Mac", "Linux", "Prediction", "Confidence"
        ])

        total_predictions = len(df)
        high_potential = len(df[df["Prediction"] == 1])
        success_rate = (high_potential / total_predictions) * 100 if total_predictions > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", total_predictions)
        col2.metric("High Potential Games", high_potential)
        col3.metric("Success Rate (%)", f"{success_rate:.2f}")

        df["Prediction"] = df["Prediction"].map({0: "Low Potential", 1: "High Potential"})
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions recorded yet.")