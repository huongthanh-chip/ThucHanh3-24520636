import numpy as np
from SVM import SVM
import cv2
import os

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "chest_xray")

def collect_data(split: str = "train"):
    normal = "NORMAL"
    pneumonia = "PNEUMONIA"

    images = []
    labels = []

    # Xử lý lớp NORMAL
    normal_path = os.path.join(BASE_DIR, split, normal)
    if not os.path.isdir(normal_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {normal_path}")

    for img_file in os.listdir(normal_path):
        img_path = os.path.join(normal_path, img_file)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (128, 128)).reshape(-1)
        images.append(img)
        labels.append(1)

    # Xử lý lớp PNEUMONIA
    pneumonia_path = os.path.join(BASE_DIR, split, pneumonia)
    if not os.path.isdir(pneumonia_path):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {pneumonia_path}")

    for img_file in os.listdir(pneumonia_path):
        img_path = os.path.join(pneumonia_path, img_file)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, (128, 128)).reshape(-1)
        images.append(img)
        labels.append(-1)

    X = np.array(images) / 255.0 # Chuẩn hóa pixel về [0, 1]
    y = np.array(labels)
    return X, y

X_train, y_train = collect_data("train")
X_test, y_test = collect_data("test")

model = SVM(learning_rate=0.0001, lambda_param=0.01, n_iters=1000)
model.fit(X_train, y_train, verbose=True)

y_predict = model.predict(X_test)
metrics = model.evaluate(y_test, y_predict)

print("\nDanh gia tren tap test:")
for metric_name, value in metrics.items():
    print(f"- {metric_name}: {value:.4f}")