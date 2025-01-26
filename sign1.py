#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import time

# Configuration
MODEL_PATH = "sign_language_model.tflite"
GEMINI_API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
CATEGORIES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
              'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit page configuration
st.set_page_config(page_title="Indian Sign Language Translator", layout="wide")

# Session state initialization
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = ""
if 'gemini_response' not in st.session_state:
    st.session_state.gemini_response = "No content generated yet."

def get_gemini_response(text):
    """Get explanation from Gemini."""
    try:
        response = genai.generate_text(
            f"Write content about the detected sign language gesture: {text}. "
            "Explain its possible meaning or context in a short paragraph."
        )
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Unable to generate content at this time."

def predict_tflite(image):
    """Perform prediction using TensorFlow Lite model."""
    # Resize and preprocess the image
    input_image = cv2.resize(image, (64, 64))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image / 255.0, axis=0).astype(np.float32)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get predicted class and confidence
    pred_class = CATEGORIES[np.argmax(output_data)]
    confidence = np.max(output_data)
    return pred_class, confidence

def process_frame(frame):
    """Crop the center of the frame and predict."""
    h, w, _ = frame.shape
    center_crop = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]

    # Predict using the cropped region
    pred_class, confidence = predict_tflite(center_crop)

    # Draw a bounding box on the frame
    cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 2)
    cv2.putText(frame, f"{pred_class} ({confidence:.2f})", (w // 4, h // 4 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, pred_class, confidence

# Streamlit UI
st.title("Indian Sign Language Translator 🤟")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, key="confidence_threshold")
    st.markdown("---")
    st.write("Last Detected Sign:")
    st.write(st.session_state.last_prediction)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam Feed")
    camera = st.camera_input("Live Feed")

with col2:
    st.subheader("Generated Content")
    st.write(st.session_state.gemini_response)

# Processing loop
if camera is not None:
    # Decode webcam feed
    bytes_data = camera.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Process frame and get predictions
    processed_frame, pred_class, confidence = process_frame(frame)

    # Update prediction and generate content if confidence is above threshold
    if confidence > confidence_threshold and pred_class != st.session_state.last_prediction:
        st.session_state.last_prediction = pred_class
        st.session_state.gemini_response = get_gemini_response(pred_class)

    # Display processed frame
    col1.image(processed_frame, channels="BGR")

