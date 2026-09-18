import streamlit as st
import numpy as np
import pickle
from PIL import Image
from xgboost import XGBClassifier


st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Waste Classification")
st.write("Upload an image to classify the type of waste.")


# Load XGBoost model
model = XGBClassifier()
model.load_model("xgb_model (1).json")


# Load scaler
with open("scaler (2).pkl", "rb") as file:
    scaler = pickle.load(file)


# Load PCA
with open("pca (1).pkl", "rb") as file:
    pca = pickle.load(file)


# Load Label Encoder
with open("label_encoder (1).pkl", "rb") as file:
    le = pickle.load(file)


# Upload image
uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    # Resize image
    image_resized = image.resize((64, 64))

    # Convert image to NumPy array
    image_array = np.array(image_resized)

    # Flatten image
    image_flatten = image_array.flatten().reshape(1, -1)

    # Apply scaler
    image_scaled = scaler.transform(image_flatten)

    # Apply PCA
    image_pca = pca.transform(image_scaled)

    # Prediction
    prediction = model.predict(image_pca)

    # Convert encoded label to original class
    predicted_class = le.inverse_transform(
        prediction.astype(int)
    )[0]

    # Display prediction
    st.success(
        f"Predicted Waste Type: {predicted_class}"
    )