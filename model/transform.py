import os
import cv2
import numpy as np
from mediapipe.tasks.python import vision
from glob import glob
import mediapipe as mp
from mediapipe.tasks import python
from tqdm import tqdm 

root_dir = "C:\\Users\\asus\\Downloads\\FingerHint\\dataset5_backup"
found_label = set()
batch = sorted([i for i in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,i))])

# ลูปแรก: ดึงชื่อ Label อย่างรวดเร็ว ไม่ต้องใช้หลอดโหลด
for partial_batch in batch:
    label_dir = os.path.join(root_dir, partial_batch) 
    label = sorted([i for i in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir,i))]) 
    found_label.update(label)

found_label = sorted(found_label)
labels_todict = {name: idx for idx,name in enumerate(found_label)}

all_x = []
all_y = []

# โหลดโมเดล
base_options = python.BaseOptions(model_asset_path='C:\\Users\\asus\\Downloads\\FingerHint\\model\\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
hands = vision.HandLandmarker.create_from_options(options)

# ลูปสอง: ประมวลผลจริง (เอา tqdm มาใส่ตรงลูปรูปภาพแทน)
for partial_batch in batch:
    label_dir = os.path.join(root_dir, partial_batch) 
    for label in found_label:
        
        img_dir = os.path.join(label_dir,label) 
        if not os.path.exists(img_dir):
            continue
        current_label = labels_todict[label]
        
        # ใส่ tqdm ตรงนี้! จะได้เห็นหลอดโหลดวิ่งทุกรูป พร้อมบอกด้วยว่าทำ Batch ไหน Label อะไรอยู่
        for img_file in tqdm(os.listdir(img_dir), desc=f"Batch {partial_batch} | Label {label}"):
            if not img_file.endswith((".png",".jpg",".jpeg")):
                continue
            
            img_path = os.path.join(img_dir,img_file)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue    
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape
            mp_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = hands.detect(mp_rgb)
            
            if results.hand_landmarks: 
                feature_vector = np.zeros(63) 
                hand_lms = results.hand_landmarks[0] 
                
                coordinate = []
                for lm in hand_lms:
                    coordinate.extend([lm.x, lm.y, lm.z])
                feature_vector[:63] = np.array(coordinate)

                
                all_x.append(feature_vector)
                all_y.append(current_label)              

hands.close()                                

output_path = "C:\\Users\\asus\\Downloads\\FingerHint\\numpy__dataset"
if not os.path.exists(output_path):
    os.makedirs(output_path)

np.save(os.path.join(output_path, "X_features.npy"), np.array(all_x, dtype=np.float32))
np.save(os.path.join(output_path, "y_labels.npy"), np.array(all_y, dtype=np.int64))
np.save(os.path.join(output_path, "classes.npy"), np.array(found_label))