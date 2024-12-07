import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import pandas as pd

# Define model paths
MODEL_PATHS = {
    "CNN": "models/cnn.h5",
    "ResNet152": "models/restnet152.h5",
    "MobileNetV3Large": "models/mobilenetv3large.keras"
}

# Load models
@st.cache_resource
def load_model_cached(model_path):
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e

# Preprocess image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)

# Predict
def predict_image(model, image):
    predictions = model.predict(image)[0]
    classes = ["Cat", "Dog", "Snake"]
    confidence_scores = {cls: float(pred) for cls, pred in zip(classes, predictions)}
    return max(confidence_scores, key=confidence_scores.get), confidence_scores


def footer():
    # Footer Section

    st.markdown('<style>div.block-container{padding-bottom: 100px;,text-align: center;}</style>', unsafe_allow_html=True)
    st.markdown("""------""")
    st.markdown("""
        <p  
        align='center'>Developed by </p>
        """, unsafe_allow_html=True)
    st.markdown("""
        <p  
        align='center'>Gandham Mani Saketh</p>
        """, unsafe_allow_html=True)

   
    st.markdown(""" <p align="center">If you want any assistances or have any  queries. just, feel free to reach out!</p>
          <p align="center">
        <a href="https://www.linkedin.com/in/gandhammanisaketh2421/" target="_blank">
            <img src="https://img.icons8.com/fluent/48/000000/linkedin.png" alt="LinkedIn" style="width:40px;"/>
        </a>
        <a href="https://github.com/GANDHAMMANI" target="_blank">
            <img src="https://img.icons8.com/fluent/48/000000/github.png" alt="GitHub" style="width:40px;"/>
        </a>
        <a href="mailto:gandhammani2421@gmail.com" target="_blank">
            <img src="https://img.icons8.com/fluent/48/000000/gmail.png" alt="GitHub" style="width:40px;"/>
        </a>
        <a href="https://www.instagram.com/mr.pandapal/">
            <img src="https://img.icons8.com/fluent/48/000000/instagram-new.png" alt="Instagram" style="width: 40px;">
        </a>
    </p>
""", unsafe_allow_html=True)
    st.markdown("""  <p align="center"> -Saketh07</p>""", unsafe_allow_html=True)


# Streamlit UI
st.set_page_config(page_title="Image Classification", page_icon="🐱🐶🐍", layout="centered")

# Styling the page
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        .title {
            font-size: 48px;
            color: white;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            padding: 20px;
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .sidebar-header {
            font-size: 20px;
            color: #fff;
            background-color: #333;
            padding: 10px;
            border-radius: 5px;
        }
        .sidebar {
            background-color: #f8f8f8;
        }
        .stImage {
            border: 3px solid #4CAF50;
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .prediction {
            font-size: 24px;
            color: white;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }
        .confidence-table {
            margin-top: 20px;
            border-collapse: collapse;
            width: 100%;
        }
        .confidence-table th, .confidence-table td {
            text-align: center;
            padding: 10px;
            border: 1px solid #ddd;
        }
        .confidence-table th {
            background-color: #4CAF50;
            color: white;
        }
        .confidence-table td {
            background-color: #f9f9f9;
        }
        .result-sentence {
            font-size: 22px;
            color: blue;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title with styling
st.markdown('<p class="title">Image Classification: Cats, Dogs, and Snakes</p>', unsafe_allow_html=True)

st.sidebar.header("Model Selection", anchor="sidebar-header")
model_choice = st.sidebar.selectbox("Choose a model:", list(MODEL_PATHS.keys()))

st.sidebar.header("Upload Image", anchor="sidebar-header")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load and display the uploaded image
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Check if the uploaded image is a valid image for classification
    image_format = image.format.lower()
    valid_formats = ["jpg", "jpeg", "png"]
    if image_format not in valid_formats:
        st.warning("This app only supports image files in JPG, JPEG, or PNG format. Please upload a valid image.")

    else:
        # Display the uploaded image with custom styling
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load the selected model
        st.write(f"Using model: {model_choice}")
        model = load_model_cached(MODEL_PATHS[model_choice])
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image, target_size=(224, 224))  # Adjust size as per model
        
        # Predict
        predicted_class, confidence_scores = predict_image(model, preprocessed_image)
        
        # Display results in a table
        st.markdown("<p class='prediction'>Predicted Class: {}</p>".format(predicted_class), unsafe_allow_html=True)
        st.markdown("<p class='confidence'>Confidence Scores:</p>", unsafe_allow_html=True)
        
        # Prepare data for the table with custom column names
        confidence_data = pd.DataFrame(list(confidence_scores.items()), columns=["Species", "Scores"])

        # Format the confidence values as percentages
        confidence_data["Scores"] = confidence_data["Scores"].apply(lambda x: f"{x:.2%}")

        # Display the confidence table with custom column names (without setting an index)
        st.table(confidence_data)

        # If one class has significantly higher confidence, display a sentence
        max_class = max(confidence_scores, key=confidence_scores.get)
        max_confidence = confidence_scores[max_class]
        
        if max_confidence > 0.75:  # Example threshold for high confidence
            st.markdown(f"<p class='result-sentence'>The image is predicted as {max_class} with high confidence!</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='result-sentence'>The model is unsure, confidence is below threshold.</p>", unsafe_allow_html=True)

else:
    st.info("This app is only for classifying Cats, Dogs, and Snakes. Please upload a valid image to classify.")

footer()