import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

def extract_file(zip_file_obj, member, extract_to_dir):
    """ฟังก์ชันย่อยสำหรับแตกไฟล์ 1 ไฟล์"""
    zip_file_obj.extract(member, extract_to_dir)

def fast_extract_zip(zip_file_path, extract_to_dir, max_workers=8):
    """
    ฟังก์ชันหลักสำหรับแตกไฟล์ด้วย Multi-threading
    max_workers: จำนวน Thread ที่ต้องการใช้ (ปรับตามจำนวน Core ของ CPU)
    """
    print(f"กำลังเริ่มต้นการแตกไฟล์จาก: {zip_file_path}")
    print(f"เป้าหมาย: {extract_to_dir}")
    print(f"จำนวน Threads ที่ใช้: {max_workers}")
    
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ ZIP ที่: {zip_file_path}")
        
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        total_files = len(members)
        print(f"พบไฟล์ทั้งหมด {total_files} ไฟล์ใน Archive")
        
        # ใช้ ThreadPoolExecutor เพื่อแตกไฟล์แบบขนาน (Parallel Extraction)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # จ่ายงานการแตกไฟล์แต่ละไฟล์เข้าไปใน Thread Pool
            for member in members:
                executor.submit(extract_file, zip_ref, member, extract_to_dir)
                
    print(f"กระบวนการแตกไฟล์แบบ Multi-threading เสร็จสิ้นทั้งหมดลงที่ {extract_to_dir}")

# ตัวอย่างการเรียกใช้งาน (ปรับ max_workers ให้เท่ากับจำนวน Logical Cores ของ CPU คุณ)
# fast_extract_zip(zip_file_path="./dataset.zip", extract_to_dir="./extracted_data", max_workers=8)

fast_extract_zip(zip_file_path="C:\\Users\\asus\\Downloads\\archive (17).zip", extract_to_dir="C:\\Users\\asus\\Downloads\\FingerHint", max_workers=8)