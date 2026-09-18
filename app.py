import streamlit as st
import numpy as np
import cv2
import pickle
import xgboost as xgb
from PIL import Image


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="wide"
)


# --------------------------------------------------
# CLASS MAPPING
# --------------------------------------------------

class_names = {
    0: "biological",
    1: "cardboard",
    2: "clothes",
    3: "green_glass",
    4: "paper",
    5: "plastic",
    6: "trash",
    7: "white_glass"
}


# --------------------------------------------------
# LOAD SCALER, PCA AND MODEL
# --------------------------------------------------

@st.cache_resource
def load_models():

    # Load StandardScaler
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    # Load PCA
    with open("pca.pkl", "rb") as file:
        pca = pickle.load(file)

    # Load XGBoost model
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")

    return scaler, pca, model


scaler, pca, model = load_models()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("♻️ Waste Classification System")

st.write(
    "Upload an image to classify the type of waste."
)


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width=400
    )


    if st.button("🔍 Predict Waste"):

        # ------------------------------------------
        # STEP 1: PIL → NumPy
        # ------------------------------------------

        image_np = np.array(image)


        # ------------------------------------------
        # STEP 2: RGB → BGR
        # ------------------------------------------

        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )


        # ------------------------------------------
        # STEP 3: Resize to 64 × 64
        # ------------------------------------------

        image_resized = cv2.resize(
            image_np,
            (64, 64),
            interpolation=cv2.INTER_LINEAR
        )


        # ------------------------------------------
        # STEP 4: Flatten
        # ------------------------------------------

        image_flat = image_resized.flatten()


        # Check number of features
        st.write(
            "Original image features:",
            image_flat.shape[0]
        )


        # ------------------------------------------
        # STEP 5: Reshape
        # ------------------------------------------

        input_data = image_flat.reshape(1, -1)


        # ------------------------------------------
        # STEP 6: StandardScaler
        # ------------------------------------------

        input_scaled = scaler.transform(input_data)


        # ------------------------------------------
        # STEP 7: PCA
        # ------------------------------------------

        input_pca = pca.transform(input_scaled)


        # Check PCA features
        st.write(
            "PCA features:",
            input_pca.shape[1]
        )


        # ------------------------------------------
        # STEP 8: XGBoost Prediction
        # ------------------------------------------

        prediction = model.predict(input_pca)


        # ------------------------------------------
        # STEP 9: Convert prediction to integer
        # ------------------------------------------

        predicted_label = int(prediction[0])


        # ------------------------------------------
        # STEP 10: Class name
        # ------------------------------------------

        predicted_class = class_names[predicted_label]


        # ------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------

        st.subheader("Prediction Result")

        st.success(
            f"♻️ Predicted Waste: **{predicted_class}**"
        )

        st.write(
            f"Class Number: **{predicted_label}**"
        )