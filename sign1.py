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
if 'buffer' not in st.session_state:
    st.session_state.buffer = []
if 'gemini_text' not in st.session_state:
    st.session_state.gemini_text = "Make gestures to begin..."
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0
if 'running' not in st.session_state:
    st.session_state.running = False

def get_gemini_response(text):
    """Get explanation from Gemini"""
    try:
        response = genai.generate_text(
            f"Interpret this sequence of Indian Sign Language gestures as meaningful text: {text}. "
            "Provide both the literal translation and possible meanings in 2 short paragraphs. "
            "If it appears to be random letters, suggest possible word formations."
        )
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Explanation unavailable"

def predict_tflite(image):
    """Perform prediction using TensorFlow Lite model"""
    # Resize and preprocess image
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
    """Process frame and classify hand gestures using basic skin-color detection."""
    # Convert the frame to HSV for skin color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for skin color (tweak these values as needed)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for skin color
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Proceed only if at least one contour is found
    if contours:
        # Find the largest contour (assuming it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the hand region
        hand_image = frame[y:y + h, x:x + w]
        
        # Only process if the region is large enough
        if hand_image.size > 0:
            pred_class, confidence = predict_tflite(hand_image)
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{pred_class} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update detection buffer
            if confidence > st.session_state.confidence_threshold:
                st.session_state.buffer.append(pred_class)
                st.session_state.buffer = st.session_state.buffer[-st.session_state.buffer_size:]

    return frame

# Streamlit UI
st.title("Indian Sign Language Translator 🤟")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    st.session_state.confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    st.session_state.buffer_size = st.slider("Buffer Size", 1, 10, 5)

    if st.button("Clear Buffer"):
        st.session_state.buffer = []
        st.session_state.gemini_text = "Buffer cleared!"

    st.markdown("---")
    st.write("Detection Buffer:")
    st.write(st.session_state.buffer)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Detection")
    run_button = st.empty()

    if st.session_state.running:
        if run_button.button("Stop Detection"):
            st.session_state.running = False
    else:
        if run_button.button("Start Detection"):
            st.session_state.running = True

    camera = st.camera_input("Webcam Feed")

with col2:
    st.subheader("Explanation")
    explanation = st.empty()
    explanation.markdown(st.session_state.gemini_text)

# Processing loop
if st.session_state.running and camera is not None:
    bytes_data = camera.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Process frame
    processed_frame = process_frame(frame)

    # Update display
    col1.image(processed_frame, channels="BGR")

    # Update Gemini text every 2 seconds
    if time.time() - st.session_state.last_update > 2 and st.session_state.buffer:
        st.session_state.gemini_text = get_gemini_response(' '.join(st.session_state.buffer))
        st.session_state.last_update = time.time()
        explanation.markdown(st.session_state.gemini_text)

