import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load the trained model
@st.cache_resource
def load_fish_model():
    model = load_model('mobilenetv2_fish_model.h5')  # Replace with your model path
    return model

model = load_fish_model()

# Define class names (edit based on your dataset)
class_names = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream','fish sea_food house_mackerel','fish sea_food red_mullet','fish sea_food red_sea_bream','fish sea_food sea_bass','fish sea_food shrimp','fish sea_food striped_red_mullet','fish sea_food trout']  # example

# App UI
st.set_page_config(page_title="🐟 Fish Classifier", layout="centered")
st.title("🐟 Fish Image Classifier")
st.markdown("Upload an image of a fish and get the predicted species.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_data = Image.open(uploaded_file).convert('RGB')
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image_data.resize((224, 224))  # Adjust size if model uses different input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if trained with normalization

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Output
    st.markdown(f"### 🐠 Predicted Fish: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence * 100:.2f}%**")
