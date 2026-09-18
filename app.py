import streamlit as st
import cv2
import numpy as np
import pickle
from xgboost import XGBClassifier


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("♻️ Waste Classification")
st.write("Upload an image to classify the type of waste.")


# --------------------------------------------------
# LOAD SAVED FILES
# --------------------------------------------------

# Load XGBoost model
model = XGBClassifier()
model.load_model("xgb_model (1).json")


# Load StandardScaler
with open("scaler (2).pkl", "rb") as file:
    scaler = pickle.load(file)


# Load PCA
with open("pca (1).pkl", "rb") as file:
    pca = pickle.load(file)


# Load Label Encoder
with open("label_encoder (1).pkl", "rb") as file:
    le = pickle.load(file)


# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    # Open image using OpenCV
    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is not None:

        # Display uploaded image
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            width="stretch"
        )

        # --------------------------------------------------
        # SAME PREPROCESSING USED DURING TRAINING
        # --------------------------------------------------

        # Resize to 64 × 64
        image = cv2.resize(
            image,
            (64, 64),
            interpolation=cv2.INTER_LINEAR
        )

        # Convert to NumPy array
        image_array = np.array(image)

        # Flatten image
        image_flatten = image_array.flatten()

        # Reshape into one sample
        image_flatten = image_flatten.reshape(1, -1)


        # --------------------------------------------------
        # STANDARD SCALER
        # --------------------------------------------------

        image_scaled = scaler.transform(
            image_flatten
        )


        # --------------------------------------------------
        # PCA
        # --------------------------------------------------

        image_pca = pca.transform(
            image_scaled
        )


        # --------------------------------------------------
        # XGBOOST PREDICTION
        # --------------------------------------------------

        prediction = model.predict(
            image_pca
        )


        # --------------------------------------------------
        # CONVERT NUMBER TO CLASS NAME
        # --------------------------------------------------

        predicted_class = le.inverse_transform(
            prediction.astype(int)
        )[0]


        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.success(
            f"Predicted Waste Type: {predicted_class}"
        )


    else:

        st.error(
            "Unable to read the uploaded image."
        )