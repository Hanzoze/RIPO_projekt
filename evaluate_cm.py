import os
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from analyze import analyze_frame, create_face_landmarker
from mediapipe.tasks.python.vision import RunningMode

BASE_PATH = "Mouth/Mouth"
CATEGORIES = ["Mild mouth", "Moderate mouth", "Moderate severe mouth", "Severe mouth"]

THRESHOLDS = {
    "Mild_to_Moderate": 0.1665,
    "Moderate_to_ModSevere": 0.1931,
    "ModSevere_to_Severe": 0.2479
}

def get_prediction(score):
    if score < THRESHOLDS["Mild_to_Moderate"]:
        return "Mild mouth"
    elif score < THRESHOLDS["Moderate_to_ModSevere"]:
        return "Moderate mouth"
    elif score < THRESHOLDS["ModSevere_to_Severe"]:
        return "Moderate severe mouth"
    else:
        return "Severe mouth"


def run_evaluation():
    landmarker = create_face_landmarker(RunningMode.IMAGE)
    y_true = []
    y_pred = []

    for cat in CATEGORIES:
        folder_path = os.path.join(BASE_PATH, cat)
        if not os.path.exists(folder_path): continue

        print(f"Testowanie kategorii: {cat}...")
        files = [f for f in os.listdir(folder_path) if f.endswith(('.bmp', '.jpg', '.png'))]

        for f in files:
            img = cv2.imread(os.path.join(folder_path, f))
            metrics = analyze_frame(img, landmarker)

            if metrics.detection_ok:
                y_true.append(cat)
                y_pred.append(get_prediction(metrics.mouth_corner_diff_norm))

    landmarker.close()

    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CATEGORIES, yticklabels=CATEGORIES, cmap='Blues')
    plt.xlabel('Predykcja (Algorytm)')
    plt.ylabel('Rzeczywistość (Etykieta)')
    plt.title('Macierz Pomyłek - Prototyp Etap 2')
    plt.savefig('confusion_matrix.png')

    print("\n=== RAPORT KLASYFIKACJI ===")
    print(classification_report(y_true, y_pred, target_names=CATEGORIES))
    print("\nWykres zapisano jako confusion_matrix.png")


if __name__ == "__main__":
    run_evaluation()