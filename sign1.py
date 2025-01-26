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
CATEGORIES = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
              'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit page configuration
st.set_page_config(page_title="Sign Language Translator", layout="wide")

# Session state initialization
if 'buffer' not in st.session_state:
    st.session_state.update({
        'buffer': [],
        'gemini_text': "Make gestures to begin...",
        'last_update': 0,
        'running': False,
        'confidence_threshold': 0.5,
        'buffer_size': 5,
        'last_predictions': [],
        'detection_zone_size': 400  # Default detection zone size
    })

def get_gemini_response(text):
    """Get explanation from Gemini"""
    try:
        response = gemini_model.generate_content(
            f"Interpret this sequence of Sign Language gestures: {text}. "
            "Provide both literal translation and possible meanings in 2 short paragraphs."
        )
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Explanation unavailable"

def process_frame(frame):
    """Process frame with fixed detection zone"""
    h, w = frame.shape[:2]
    current_predictions = []
    
    # Create detection zone rectangle (centered)
    box_size = st.session_state.detection_zone_size
    x_min = (w - box_size) // 2
    y_min = (h - box_size) // 2
    x_max = x_min + box_size
    y_max = y_min + box_size
    
    # Draw detection zone
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(frame, "Place hand here", (x_min + 10, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Extract and process hand region
    hand_image = frame[y_min:y_max, x_min:x_max]
    
    if hand_image.size == 0:
        return frame

    # Preprocessing
    processed_image = cv2.resize(hand_image, (64, 64))
    processed_image = processed_image.astype(np.float32) / 255.0
    processed_image = np.expand_dims(processed_image, axis=0)

    # Model inference
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    
    pred_class = CATEGORIES[np.argmax(pred)]
    confidence = np.max(pred)

    # Temporal filtering
    st.session_state.last_predictions.append(pred_class)
    st.session_state.last_predictions = st.session_state.last_predictions[-3:]
    
    if confidence > st.session_state.confidence_threshold:
        if len(set(st.session_state.last_predictions)) == 1:
            cv2.putText(frame, f"{pred_class} ({confidence:.2f})",
                      (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            current_predictions.append(pred_class)

    # Update detection buffer
    if current_predictions:
        st.session_state.buffer.extend(current_predictions)
        st.session_state.buffer = st.session_state.buffer[-st.session_state.buffer_size:]
    
    return frame

# Streamlit UI
st.title("Sign Language Translator 🤟")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    st.session_state.confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.session_state.buffer_size = st.slider("Buffer Size", 1, 20, 5)
    st.session_state.detection_zone_size = st.slider("Detection Zone Size", 200, 600, 400)
    
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

    # Dynamic camera refresh
    if st.session_state.running:
        camera = st.camera_input("Webcam Feed", key=f"cam_{time.time()}")
    else:
        camera = None

with col2:
    st.subheader("Explanation")
    explanation = st.empty()
    explanation.markdown(st.session_state.gemini_text)

# Processing loop
if st.session_state.running and camera is not None:
    bytes_data = camera.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    processed_frame = process_frame(frame)
    col1.image(processed_frame, channels="BGR")
    
    # Update Gemini explanation every 3 seconds
    if time.time() - st.session_state.last_update > 3 and st.session_state.buffer:
        st.session_state.gemini_text = get_gemini_response(' '.join(st.session_state.buffer))
        st.session_state.last_update = time.time()
        explanation.markdown(st.session_state.gemini_text)

