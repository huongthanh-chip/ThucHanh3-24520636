import os
from typing import Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "chest_xray")
CLASS_LABELS = {
    "NORMAL": 1,
    "PNEUMONIA": -1,
}


def load_chest_xray_data(
    split: str,
    image_size: Tuple[int, int] = (128, 128),
    max_per_class: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Chest X-Ray images as flattened grayscale vectors."""
    images = []
    labels = []

    for class_name, label in CLASS_LABELS.items():
        class_dir = os.path.join(BASE_DIR, split, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Data folder not found: {class_dir}")

        image_files = sorted(
            file_name
            for file_name in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, file_name))
        )
        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, image_size).reshape(-1)
            images.append(image)
            labels.append(label)

    X = np.asarray(images, dtype=np.float32) / 255.0
    y = np.asarray(labels, dtype=np.int32)
    return X, y


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    for metric_name, value in metrics.items():
        print(f"- {metric_name}: {value:.4f}")
