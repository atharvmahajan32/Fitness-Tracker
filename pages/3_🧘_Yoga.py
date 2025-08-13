import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import datetime
import os
# from playsound import playsound  # can't play sound to user in cloud, optional

# ------------------ Utility Functions ------------------ #
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Global pose timing counters
last_second = 0
counter = 0
pose_number = 1

def count_time(time_interval):
    """Handles tracking of time per pose and moving to next pose."""
    global last_second, counter, pose_number
    now = datetime.datetime.now()
    current_second = int(now.strftime("%S"))
    if current_second != last_second:
        last_second = current_second
        counter += 1
        if counter == time_interval + 1:
            counter = 0
            pose_number += 1
            # playsound('bell.wav')  # optional
            if pose_number > 3:  # last pose number per track
                pose_number = 1
    return counter, pose_number

# ------------------ Mediapipe Setup ------------------ #
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="Yoga Pose Tracker", layout="wide")
st.title("🧘 Yoga Pose Tracker - Real Time")

# Load pose images
img1 = Image.open("gif/yoga.gif")
img2 = Image.open("images/pranamasana2.png")
img3 = Image.open("images/Eka_Pada_Pranamasana.png")
img4 = Image.open("images/Ashwa_Sanchalanasana.webp")
img5 = Image.open("images/ardha_chakrasana.webp")
img6 = Image.open("images/Utkatasana.png")
img7 = Image.open("images/Veerabhadrasan_2.png")

mode = st.sidebar.selectbox("Choose the exercise", ["About","Track 1","Track 2"])

if mode == "About":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## Welcome to the Yoga arena")
        st.write("""
        Instructions:
        - Make sure your webcam is accessible.
        - Works on Streamlit Cloud and locally using browser webcam.
        - Ensure proper lighting and background.
        - One person at a time.
        """)
    with col2:
        st.image(img1, width=400)

# ------------------ Video Processor ------------------ #
class YogaVideoProcessor(VideoProcessorBase):
    def __init__(self, track_id):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.track_id = track_id

    def recv(self, frame):
        global counter, pose_number
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            )
            # Extract landmarks
            lm = results.pose_landmarks.landmark
            # Convert to pixel coords
            def get_point(name):
                p = lm[name.value]
                return [p.x * w, p.y * h]
            l_sh, r_sh = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER), get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            l_wr, r_wr = get_point(mp_pose.PoseLandmark.LEFT_WRIST), get_point(mp_pose.PoseLandmark.RIGHT_WRIST)
            l_hp, r_hp = get_point(mp_pose.PoseLandmark.LEFT_HIP), get_point(mp_pose.PoseLandmark.RIGHT_HIP)
            l_el, r_el = get_point(mp_pose.PoseLandmark.LEFT_ELBOW), get_point(mp_pose.PoseLandmark.RIGHT_ELBOW)
            l_kn, r_kn = get_point(mp_pose.PoseLandmark.LEFT_KNEE), get_point(mp_pose.PoseLandmark.RIGHT_KNEE)
            l_an, r_an = get_point(mp_pose.PoseLandmark.LEFT_ANKLE), get_point(mp_pose.PoseLandmark.RIGHT_ANKLE)

            # --- Pose Checking Logic from your original code ---
            if self.track_id == 1:
                if pose_number == 1: # Pranamasana
                    l_angle = calculate_angle(l_wr, l_sh, l_hp)
                    r_angle = calculate_angle(r_wr, r_sh, r_hp)
                    dist = np.linalg.norm(np.array(r_wr) - np.array(l_wr)) / w
                    if l_angle < 100 and r_angle < 100 and dist < 0.1:
                        cv2.putText(img, "Pose: Correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        counter, pose_number = count_time(5)
                        cv2.putText(img, f"TIME: {counter}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    else:
                        cv2.putText(img, "Pose: Incorrect", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        counter = 0
                elif pose_number == 2: # Eka Pada
                    l_angle = calculate_angle(l_wr, l_sh, l_hp)
                    r_angle = calculate_angle(r_wr, r_sh, r_hp)
                    r_knee_angle = calculate_angle(r_hp, r_kn, r_an)
                    dist = np.linalg.norm(np.array(r_wr) - np.array(l_wr)) / w
                    if l_angle > 100 and r_angle > 100 and r_knee_angle < 90 and dist < 0.1:
                        cv2.putText(img, "asana: Correct", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        counter, pose_number = count_time(5)
                    else:
                        cv2.putText(img, "Pose: Incorrect", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        counter = 0
                elif pose_number == 3: # Ashwa Sanchalanasana
                    l_leg_angle = calculate_angle(l_hp, l_kn, l_an)
                    r_leg_angle = calculate_angle(r_hp, r_kn, r_an)
                    if l_leg_angle > 90 and r_leg_angle < 150:
                        cv2.putText(img, "asana: Correct", (50, 50), 1, (0, 255, 0), 2)
                        counter, pose_number = count_time(5)
                    else:
                        cv2.putText(img, "asana: Incorrect", (50, 50), 1, (0, 0, 255), 2)
                        counter = 0

            elif self.track_id == 2:
                if pose_number == 1: # Ardha Chakrasana
                    l_angle = calculate_angle(l_wr, l_sh, l_hp)
                    r_angle = calculate_angle(r_wr, r_sh, r_hp)
                    dist = np.linalg.norm(np.array(r_wr) - np.array(l_wr)) / w
                    if l_angle > 100 and r_angle > 100 and dist < 0.1:
                        cv2.putText(img, "Pose: Correct", (50, 50), 1, (0, 255, 0), 2)
                        counter, pose_number = count_time(5)
                    else:
                        cv2.putText(img, "Pose: Incorrect", (50, 50), 1, (0, 0, 255), 2)
                        counter = 0
                elif pose_number == 2: # Utkatasana
                    l_leg_angle = calculate_angle(l_hp, l_kn, l_an)
                    r_leg_angle = calculate_angle(r_hp, r_kn, r_an)
                    if r_leg_angle < 150 and l_leg_angle < 150:
                        cv2.putText(img, "asana: Correct", (50, 50), 1, (0, 255, 0), 2)
                        counter, pose_number = count_time(5)
                    else:
                        cv2.putText(img, "asana: Incorrect", (50, 50), 1, (0, 0, 255), 2)
                        counter = 0
                elif pose_number == 3: # Veerabhadrasana 2
                    r_leg_angle = calculate_angle(r_hp, r_kn, r_an)
                    if r_leg_angle < 120:
                        cv2.putText(img, "asana: Correct", (50, 50), 1, (0, 255, 0), 2)
                        counter, pose_number = count_time(5)
                    else:
                        cv2.putText(img, "asana: Incorrect", (50, 50), 1, (0, 0, 255), 2)
                        counter = 0

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------ Run Track ------------------ #
if mode == "Track 1":
    st.subheader("Track 1: Pranamasana → Eka Pada ∶ Ashwa Sanchalanasana")
    webrtc_streamer(key="track1", video_processor_factory=lambda: YogaVideoProcessor(track_id=1),
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)

elif mode == "Track 2":
    st.subheader("Track 2: Ardha Chakrasana → Utkatasana → Veerabhadrasana 2")
    webrtc_streamer(key="track2", video_processor_factory=lambda: YogaVideoProcessor(track_id=2),
                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)
