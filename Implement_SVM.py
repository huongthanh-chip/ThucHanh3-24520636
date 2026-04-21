import numpy as np
from SVM import SVM
import cv2
import os

BASE_DIR = "data/chest_ray"

def collect_data(split: str = "train"):
    normal = "NORMAL"
    pneumonia = "PNEUMONIA"

    images = []
    labels = []
    
    # Xử lý lớp NORMAL
    normal_path = os.path.join(BASE_DIR, split, normal)
    for img_file in os.listdir(normal_path):
        img = cv2.imread(os.path.join(normal_path, img_file)) # Sửa: imread
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Sửa: BGR2GRAY
        img = cv2.resize(img, (128, 128)).reshape(-1) # Sửa: resize
        images.append(img)
        labels.append(1)

    # Xử lý lớp PNEUMONIA
    pneumonia_path = os.path.join(BASE_DIR, split, pneumonia)
    for img_file in os.listdir(pneumonia_path):
        img = cv2.imread(os.path.join(pneumonia_path, img_file)) # Sửa: imread
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128)).reshape(-1)
        images.append(img)
        labels.append(-1)

    X = np.array(images) / 255.0 # Chuẩn hóa pixel về [0, 1]
    y = np.array(labels)
    return X, y