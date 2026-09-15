import streamlit as st
import requests


st.set_page_config(
    page_title="Customer Support Classifier",
    page_icon="💬"
)


st.title("Customer Support Intent Classifier")

st.write(
    "Enter a customer support message and the ML model "
    "will predict the customer's intent."
)


message = st.text_area(
    "Customer Message",
    placeholder="Example: I forgot my password and cannot access my account"
)


if st.button("Predict Intent"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={
                "message": message
            }
        )

        if response.status_code == 200:

            result = response.json()

            st.success(
                f"Predicted Intent: {result['intent']}"
            )

        else:

            st.error("Something went wrong with the API.")