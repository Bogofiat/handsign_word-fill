import os 
from tqdm import tqdm 

def rename_folder(path):
    for folder in tqdm(os.listdir(path),desc="Changing foldername"):
        if folder == folder.lower():
            continue
        else:
            new_name = folder.lower()
            os.rename(os.path.join(path, folder), os.path.join(path, new_name))


rename_folder(r"C:\Users\asus\Downloads\Space_zer\asl_alphabet_train\asl_alphabet_train")