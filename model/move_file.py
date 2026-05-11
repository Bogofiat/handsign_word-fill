#ไฟล์นี้ใช้สำหรับ นำข้อมูลของ asl_alphabet มาเสริมกับ dataset ดังเดิม
import os 
import shutil
import numpy as np 
from tqdm import tqdm


def split_dataset(start,destination,target_dirs=("A","B","C","D","E"),seed=42):
    rng = np.random.default_rng(seed)
    #วนลูปแต่ละคลาส a,b,c,d,e,f,g,... 
    for cls in tqdm(os.listdir(start),desc="Splitting dataset"):
        src_cls = os.path.join(start,cls) #ต่อpath \dataset\A
        if not os.path.isdir(src_cls):
            continue

        valid_target = [target for target in target_dirs if os.path.isdir(os.path.join(destination,target,cls))]
        if not valid_target:
            print(f"ไม่มีโฟลเดอร์เป้าหมายสำหรับคลาส {cls} ใน {destination}")
            continue
        
        #เลือกไฟล์ทั้งหมดในคลาสนั้นๆ แล้วสุ่มแบ่งเป็นกลุ่มๆ ตามจำนวนโฟลเดอร์เป้าหมายที่มีอยู่
        files = [f for f in os.listdir(src_cls) 
                 if os.path.isfile(os.path.join(src_cls, f))]
        rng.shuffle(files)

        chunks = np.array_split(files, len(valid_target))

        for tgt, chunk in zip(valid_target, chunks):
            dst_cls = os.path.join(destination, tgt, cls) #ต่อpath \dataset5_backup\A\a
            # วนคัดลอกแต่ละไฟล์ใน chunk แต่ละ chunk คือไฟล์ที่ถูกสุ่มมาแล้วสำหรับโฟลเดอร์เป้าหมายแต่ละอัน
            for file in chunk:
                src_file = os.path.join(src_cls, file)
                dst_file = os.path.join(dst_cls, file)
                shutil.copy2(src_file, dst_file)

source = r"C:\Users\asus\Downloads\Space_zer\asl_alphabet_train\asl_alphabet_train"
target = r"C:\Users\asus\Downloads\FingerHint\dataset5_backup"

split_dataset(source, target)


