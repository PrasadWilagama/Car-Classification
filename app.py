import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt

# Load the model
loaded_model = load_model('advanced_car_classification_model_md.h5', compile=False)

# Define car classes
car_classes = ["alto_modified", "civic_modified", "wagonr_modified"]

# Create file uploader widget
st.title("Advanced Modified Car Classification")
st.write("Upload an image and let the model predict the car class.")

uploader = st.file_uploader(
    label="Upload an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

# Define prediction function
def predict_car(image_data):
    img = Image.open(image_data)
    img = img.resize((150, 150))  # Resize to 150x150
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = loaded_model.predict(img_array)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_car_class = car_classes[predicted_class_index]
    
    return predicted_car_class, predictions

# Display uploaded image and prediction
if uploader:
    st.subheader("Uploaded Image")
    uploaded_image = uploader.read()
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    predicted_car, prediction_probs = predict_car(io.BytesIO(uploaded_image))
    
    st.subheader("Prediction")
    st.write(f"Predicted Car: {predicted_car}")

    st.subheader("Prediction Probabilities")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Alto Modified")
        st.write("Civic Modified")
    with col2:
        st.write(f"{prediction_probs[0]:.2%}")
        st.write(f"{prediction_probs[1]:.2%}")

    st.subheader("Prediction Probability Chart")
    # Create a simple pie chart using Matplotlib
    fig, ax = plt.subplots()
    ax.pie(prediction_probs, labels=car_classes, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    st.pyplot(fig)
