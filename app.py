import joblib
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI(
    title="Customer Support Intent Classifier",
    description="API for classifying customer support messages",
    version="1.0"
)


# Load trained model
model = joblib.load("intent_classifier.pkl")


# Request format
class Message(BaseModel):
    message: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Support Classifier is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: Message):

    prediction = model.predict([
        data.message
    ])

    return {
        "message": data.message,
        "intent": prediction[0]
    }