# Import libraries
import streamlit as st
import pandas as pd
import joblib


# Configure the page
st.set_page_config(
    page_title="Delhi House Price Prediction",
    page_icon="🏠"
)


# Load the cleaned dataset
df = pd.read_csv("MagicBricks_cleaned.csv")


# Load the saved model pipeline
model = joblib.load(
    "model/house_price_model.joblib"
)


# Application heading
st.title("🏠 Delhi House Price Prediction")

st.write(
    "Enter the property information below "
    "to predict its estimated price."
)


# Show basic dataset information
st.subheader("Dataset Information")

column1, column2 = st.columns(2)

column1.metric(
    "Number of Properties",
    df.shape[0]
)

column2.metric(
    "Number of Features",
    9
)


# Display the dataset inside an expandable section
with st.expander("View Cleaned Dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# Input section
st.subheader("Enter House Details")


# Numerical inputs
area = st.number_input(
    "Area in square feet",
    min_value=50.0,
    max_value=25000.0,
    value=1200.0
)

bhk = st.number_input(
    "Number of BHK",
    min_value=1,
    max_value=10,
    value=3
)

bathroom = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=7,
    value=2
)

parking = st.number_input(
    "Number of Parking Spaces",
    min_value=1,
    max_value=5,
    value=1
)


# Categorical inputs
furnishing = st.selectbox(
    "Furnishing",
    sorted(df["Furnishing"].unique())
)

locality = st.selectbox(
    "Locality",
    sorted(df["Locality"].unique())
)

status = st.selectbox(
    "Property Status",
    sorted(df["Status"].unique())
)

transaction = st.selectbox(
    "Transaction Type",
    sorted(df["Transaction"].unique())
)

property_type = st.selectbox(
    "Property Type",
    sorted(df["Type"].unique())
)


# Prediction button
if st.button("Predict House Price"):

    # Arrange inputs in the same format used during training
    input_data = pd.DataFrame(
        {
            "Area": [area],
            "BHK": [bhk],
            "Bathroom": [bathroom],
            "Furnishing": [furnishing],
            "Locality": [locality],
            "Parking": [parking],
            "Status": [status],
            "Transaction": [transaction],
            "Type": [property_type]
        }
    )


    # Predict the price
    predicted_price = model.predict(
        input_data
    )[0]


    # Display the price in crore or lakh
    if predicted_price >= 10000000:

        price_result = (
            predicted_price / 10000000
        )

        st.success(
            f"Estimated House Price: ₹{price_result:.2f} crore"
        )

    else:

        price_result = (
            predicted_price / 100000
        )

        st.success(
            f"Estimated House Price: ₹{price_result:.2f} lakh"
        )


    st.caption(
        "This is a machine-learning estimate "
        "and not an official property valuation."
    )