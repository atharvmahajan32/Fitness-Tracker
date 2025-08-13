import datetime
import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
from playsound import playsound
import os

# ----------------------
# Utility Functions
# ----------------------
def calculate_angle(a, b, c):
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle

def count_time(time_interval):
    global last_second, counter, pose_number
    now = datetime.datetime.now()
    current_second = int(now.strftime("%S"))
    if current_second != last_second:
        last_second = current_second
        counter += 1
        if counter == time_interval + 1:
            counter = 0
            pose_number += 1
            playsound('bell.wav')
            if pose_number == 5:
                pose_number = 1
    return counter, pose_number

# ----------------------
# Globals
# ----------------------
last_second = 0
counter = 0
pose_number = 1

# Load images (make sure files exist)
img1 = Image.open("gif/yoga.gif")
img2 = Image.open("images/pranamasana2.png")
img3 = Image.open("images/Eka_Pada_Pranamasana.png")
img4 = Image.open("images/Ashwa_Sanchalanasana.webp")
img5 = Image.open("images/ardha_chakrasana.webp")
img6 = Image.open("images/Utkatasana.png")
img7 = Image.open("images/Veerabhadrasan_2.png")

# Setup Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ----------------------
# Streamlit UI
# ----------------------
st.title("Yoga Pose Tracker")

app_mode = st.sidebar.selectbox("Choose the exercise", ["About", "Track 1", "Track 2"])

if app_mode == "About":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## Welcome to the Yoga arena")
        st.markdown("Choose the Track from the left sidebar")
        st.write("""
        General instructions:
        - **Webcam required** for local run.
        - **Streamlit Cloud**: live webcam streaming is not supported; you'll use camera snapshots.
        - Use a well-lit space.
        """)
    with col2:
        st.image(img1, width=400)

# -------------
# Helper: Process Frame
# -------------
def process_frame(image):
    global counter, pose_number
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
    return image

# ----------------------
# TRACK 1 and TRACK 2 Logic
# ----------------------
if app_mode in ["Track 1", "Track 2"]:
    st.subheader(f"Welcome to {app_mode}")
    use_live_cam = st.checkbox("Use live webcam (local only)", value=False)

    if use_live_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access webcam. "
                     "This feature won't work on Streamlit Cloud.")
        else:
            start = st.button("Start")
            stop = st.button("Stop")
            FRAME_WINDOW = st.empty()
            while start and not stop:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.error("Failed to grab frame from webcam.")
                    break
                frame = process_frame(frame)
                FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
            cap.release()

    else:
        # Fallback for Streamlit Cloud: Camera snapshot
        img_file = st.camera_input("Take a picture to analyze pose")
        if img_file:
            image = np.array(Image.open(img_file))
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            processed = process_frame(image_bgr)
            st.image(processed, channels="BGR", use_container_width=True)
