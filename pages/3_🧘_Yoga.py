import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import datetime

# ====== Utility Functions ======
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

last_second = 0
counter = 0
pose_number = 1

def count_time(time_interval, max_pose):
    """Track how long current pose is held and move to next."""
    global last_second, counter, pose_number
    now = datetime.datetime.now()
    current_second = int(now.strftime("%S"))
    if current_second != last_second:
        last_second = current_second
        counter += 1
        if counter == time_interval + 1:
            counter = 0
            pose_number += 1
            if pose_number > max_pose:
                pose_number = 1
    return counter, pose_number

# ===== Mediapipe Setup =====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ===== Streamlit Layout =====
st.set_page_config(page_title="Yoga Pose Tracker", layout="wide")
st.title("🧘 Yoga Pose Tracker - Real Time")

# Load images
img1 = Image.open("gif/yoga.gif")
img2 = Image.open("images/pranamasana2.png")
img3 = Image.open("images/Eka_Pada_Pranamasana.png")
img4 = Image.open("images/Ashwa_Sanchalanasana.webp")
img5 = Image.open("images/ardha_chakrasana.webp")
img6 = Image.open("images/Utkatasana.png")
img7 = Image.open("images/Veerabhadrasan_2.png")

mode = st.sidebar.selectbox("Choose the exercise", ["About", "Track 1", "Track 2"])

if mode == "About":
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("## Welcome to the Yoga Arena")
        st.write("""
        - Works in browser — no local camera drivers needed.
        - Best in a well-lit, clear background.
        - One person at a time for tracking.
        """)
    with c2:
        st.image(img1, width=400)

# ===== Video Processor =====
class YogaVideoProcessor(VideoProcessorBase):
    def __init__(self, track_id):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.track_id = track_id

    def recv(self, frame):
        global counter, pose_number
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
            )
            lm = results.pose_landmarks.landmark
            def pt(name): return [lm[name.value].x * w, lm[name.value].y * h]

            l_sh, r_sh = pt(mp_pose.PoseLandmark.LEFT_SHOULDER), pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            l_wr, r_wr = pt(mp_pose.PoseLandmark.LEFT_WRIST), pt(mp_pose.PoseLandmark.RIGHT_WRIST)
            l_hp, r_hp = pt(mp_pose.PoseLandmark.LEFT_HIP), pt(mp_pose.PoseLandmark.RIGHT_HIP)
            l_el, r_el = pt(mp_pose.PoseLandmark.LEFT_ELBOW), pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
            l_kn, r_kn = pt(mp_pose.PoseLandmark.LEFT_KNEE), pt(mp_pose.PoseLandmark.RIGHT_KNEE)
            l_an, r_an = pt(mp_pose.PoseLandmark.LEFT_ANKLE), pt(mp_pose.PoseLandmark.RIGHT_ANKLE)

            # --- Track 1 logic ---
            if self.track_id == 1:
                if pose_number == 1:  # Pranamasana
                    la, ra = calculate_angle(l_wr, l_sh, l_hp), calculate_angle(r_wr, r_sh, r_hp)
                    dist = np.linalg.norm(np.array(r_wr)-np.array(l_wr))/w
                    if la < 100 and ra < 100 and dist < 0.1:
                        cv2.putText(image, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                        counter, pose_number = count_time(5, 3)
                    else:
                        cv2.putText(image, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2); counter=0
                elif pose_number == 2:  # Eka Pada
                    la, ra = calculate_angle(l_wr, l_sh, l_hp), calculate_angle(r_wr, r_sh, r_hp)
                    rknee_a = calculate_angle(r_hp, r_kn, r_an)
                    dist = np.linalg.norm(np.array(r_wr)-np.array(l_wr))/w
                    if la > 100 and ra > 100 and rknee_a < 90 and dist < 0.1:
                        cv2.putText(image, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                        counter, pose_number = count_time(5, 3)
                    else:
                        cv2.putText(image, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2); counter=0
                elif pose_number == 3:  # Ashwa Sanchalanasana
                    lleg_a, rleg_a = calculate_angle(l_hp, l_kn, l_an), calculate_angle(r_hp, r_kn, r_an)
                    if lleg_a > 90 and rleg_a < 150:
                        cv2.putText(image, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                        counter, pose_number = count_time(5, 3)
                    else:
                        cv2.putText(image, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2); counter=0

            # --- Track 2 logic ---
            elif self.track_id == 2:
                if pose_number == 1:  # Ardha Chakrasana
                    la, ra = calculate_angle(l_wr, l_sh, l_hp), calculate_angle(r_wr, r_sh, r_hp)
                    dist = np.linalg.norm(np.array(r_wr)-np.array(l_wr))/w
                    if la > 100 and ra > 100 and dist < 0.1:
                        cv2.putText(image, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                        counter, pose_number = count_time(5, 3)
                    else:
                        cv2.putText(image, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2); counter=0
                elif pose_number == 2:  # Utkatasana
                    lleg_a, rleg_a = calculate_angle(l_hp, l_kn, l_an), calculate_angle(r_hp, r_kn, r_an)
                    if lleg_a < 150 and rleg_a < 150:
                        cv2.putText(image, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                        counter, pose_number = count_time(5, 3)
                    else:
                        cv2.putText(image, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2); counter=0
                elif pose_number == 3:  # Veerabhadrasana 2
                    rleg_a = calculate_angle(r_hp, r_kn, r_an)
                    if rleg_a < 120:
                        cv2.putText(image, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                        counter, pose_number = count_time(5, 3)
                    else:
                        cv2.putText(image, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2); counter=0

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# ===== Run Track Pages =====
if mode == "Track 1":
    st.image(img2, width=200); st.image(img3, width=200); st.image(img4, width=200)
    webrtc_streamer(
        key="track1",
        video_processor_factory=lambda: YogaVideoProcessor(track_id=1),
        rtc_configuration={  # STUN/TURN for better connectivity
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                # Uncomment & set your TURN if needed:
                # {"urls": ["turn:turnserver.example.org"], "username": "user", "credential": "pass"}
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif mode == "Track 2":
    st.image(img5, width=200); st.image(img6, width=200); st.image(img7, width=200)
    webrtc_streamer(
        key="track2",
        video_processor_factory=lambda: YogaVideoProcessor(track_id=2),
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
