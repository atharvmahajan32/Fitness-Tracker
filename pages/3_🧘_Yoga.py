import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import datetime
from PIL import Image

# ===== Utility Functions =====
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

last_second = 0
counter = 0
pose_number = 1

def count_time(time_interval, max_pose):
    global last_second, counter, pose_number
    now = datetime.datetime.now()
    current_second = int(now.strftime("%S"))
    if current_second != last_second:
        last_second = current_second
        counter += 1
        if counter > time_interval:
            counter = 0
            pose_number += 1
            if pose_number > max_pose:
                pose_number = 1
    return counter, pose_number

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ===== Streamlit Layout =====
st.set_page_config(page_title="Local Yoga Pose Tracker", layout="wide")
st.title("🧘 Yoga Pose Tracker (Local Webcam)")

# Load your pose images
img1 = Image.open("gif/yoga.gif")
img2 = Image.open("images/pranamasana2.png")
img3 = Image.open("images/Eka_Pada_Pranamasana.png")
img4 = Image.open("images/Ashwa_Sanchalanasana.webp")
img5 = Image.open("images/ardha_chakrasana.webp")
img6 = Image.open("images/Utkatasana.png")
img7 = Image.open("images/Veerabhadrasan_2.png")

mode = st.sidebar.selectbox("Choose the exercise", ["About", "Track 1", "Track 2"])

if mode == "About":
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        This version works only on your local machine.
        - Requires a connected webcam.
        - Run with `streamlit run yoga_app.py`.
        """)
    with col2:
        st.image(img1, width=400)

elif mode in ["Track 1", "Track 2"]:
    start = st.button("Start Webcam")
    stop = st.button("Stop")
    FRAME_WINDOW = st.image([])

    if start and not stop:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam not accessible!")
        else:
            max_pose = 3
            global pose_number
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame.")
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                    )
                    lm = results.pose_landmarks.landmark
                    h, w, _ = frame.shape
                    def pt(lm_name): 
                        p = lm[lm_name.value]
                        return [p.x * w, p.y * h]

                    l_sh, r_sh = pt(mp_pose.PoseLandmark.LEFT_SHOULDER), pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                    l_wr, r_wr = pt(mp_pose.PoseLandmark.LEFT_WRIST), pt(mp_pose.PoseLandmark.RIGHT_WRIST)
                    l_hp, r_hp = pt(mp_pose.PoseLandmark.LEFT_HIP), pt(mp_pose.PoseLandmark.RIGHT_HIP)
                    l_kn, r_kn = pt(mp_pose.PoseLandmark.LEFT_KNEE), pt(mp_pose.PoseLandmark.RIGHT_KNEE)
                    l_an, r_an = pt(mp_pose.PoseLandmark.LEFT_ANKLE), pt(mp_pose.PoseLandmark.RIGHT_ANKLE)

                    if mode == "Track 1":
                        if pose_number == 1: # Pranamasana
                            la, ra = calculate_angle(l_wr, l_sh, l_hp), calculate_angle(r_wr, r_sh, r_hp)
                            dist = np.linalg.norm(np.array(r_wr)-np.array(l_wr))/w
                            if la < 100 and ra < 100 and dist < 0.1:
                                cv2.putText(frame, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                                counter, pose_number = count_time(5, max_pose)
                            else:
                                cv2.putText(frame, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2) 
                                counter = 0
                        elif pose_number == 2: # Eka Pada
                            la, ra = calculate_angle(l_wr, l_sh, l_hp), calculate_angle(r_wr, r_sh, r_hp)
                            rknee_a = calculate_angle(r_hp, r_kn, r_an)
                            dist = np.linalg.norm(np.array(r_wr)-np.array(l_wr))/w
                            if la > 100 and ra > 100 and rknee_a < 90 and dist < 0.1:
                                cv2.putText(frame, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                                counter, pose_number = count_time(5, max_pose)
                            else:
                                cv2.putText(frame, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2)
                                counter=0
                        elif pose_number == 3: # Ashwa
                            lleg_a = calculate_angle(l_hp, l_kn, l_an)
                            rleg_a = calculate_angle(r_hp, r_kn, r_an)
                            if lleg_a > 90 and rleg_a < 150:
                                cv2.putText(frame, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                                counter, pose_number = count_time(5, max_pose)
                            else:
                                cv2.putText(frame, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2)
                                counter=0

                    elif mode == "Track 2":
                        if pose_number == 1: # Ardha Chakrasana
                            la, ra = calculate_angle(l_wr, l_sh, l_hp), calculate_angle(r_wr, r_sh, r_hp)
                            dist = np.linalg.norm(np.array(r_wr)-np.array(l_wr))/w
                            if la > 100 and ra > 100 and dist < 0.1:
                                cv2.putText(frame, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                                counter, pose_number = count_time(5, max_pose)
                            else:
                                cv2.putText(frame, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2)
                                counter=0
                        elif pose_number == 2: # Utkatasana
                            lleg_a = calculate_angle(l_hp, l_kn, l_an)
                            rleg_a = calculate_angle(r_hp, r_kn, r_an)
                            if lleg_a < 150 and rleg_a < 150:
                                cv2.putText(frame, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                                counter, pose_number = count_time(5, max_pose)
                            else:
                                cv2.putText(frame, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2)
                                counter=0
                        elif pose_number == 3: # Veerabhadrasana 2
                            rleg_a = calculate_angle(r_hp, r_kn, r_an)
                            if rleg_a < 120:
                                cv2.putText(frame, "Pose Correct", (50,50), 1, 1.5, (0,255,0), 2)
                                counter, pose_number = count_time(5, max_pose)
                            else:
                                cv2.putText(frame, "Pose Incorrect", (50,50), 1, 1.5, (0,0,255), 2)
                                counter=0

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
