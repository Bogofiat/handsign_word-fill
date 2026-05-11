import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os, time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# 1) โหลดโมเดล NN + ชื่อคลาส
numpy_files = "C:\\Users\\asus\\Downloads\\FingerHint\\numpy__dataset"
model = tf.keras.models.load_model("finger_hint_model_new_more_data.h5")
Class = np.load(os.path.join(numpy_files, "classes.npy"))

# 2) ตั้ง HandLandmarker (Tasks API ใหม่)
base_opts = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_opts,
    running_mode=vision.RunningMode.VIDEO,    # โหมด video (sync, ใช้ timestamp)
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# helper วาด landmark (API ใหม่ต้องแปลงกลับเป็น proto ก่อน)
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

def draw_hand(frame, hand_lm):
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_lm
    ])
    mp_draw.draw_landmarks(
        frame, proto, HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style()
    )

# 3) เปิดกล้อง
cap = cv2.VideoCapture(0)
t0 = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # แปลงเป็น mp.Image แล้วส่งเข้า detector พร้อม timestamp(ms)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int((time.time() - t0) * 1000)
    result = landmarker.detect_for_video(mp_image, ts_ms)

    label_text = "No hand"

    if result.hand_landmarks:
        hand_lm = result.hand_landmarks[0]               # มือแรก
        draw_hand(frame, hand_lm)

        # flatten 21 จุด -> 63 ค่า (ลำดับเดียวกับตอนเทรน)
        feats = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm]).flatten()
        feats = feats.reshape(1, -1)

        prob = model.predict(feats, verbose=0)[0]
        idx  = np.argmax(prob)
        label_text = f"{Class[idx]}  ({prob[idx]*100:.1f}%)"

    cv2.putText(frame, label_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("FingerHint - Live", frame)

    if cv2.waitKey(1) & 0xFF == 27:                       # ESC ออก
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()