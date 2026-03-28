import os
import cv2
import pandas as pd
from analyze import analyze_frame, create_face_landmarker
from mediapipe.tasks.python.vision import RunningMode

BASE_PATH = "Mouth/Mouth"
CATEGORIES = ["Mild mouth", "Moderate mouth", "Moderate severe mouth", "Severe mouth"]

def run_calibration():
    landmarker = create_face_landmarker(RunningMode.IMAGE)
    all_data = []

    for cat in CATEGORIES:
        folder_path = os.path.join(BASE_PATH, cat)
        if not os.path.exists(folder_path):
            print(f"[WARN] Brak folderu: {folder_path}")
            continue

        print(f"Analizuję kategorię: {cat}...")
        files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.bmp', '.jpeg'))]

        for f in files:
            img = cv2.imread(os.path.join(folder_path, f))
            metrics = analyze_frame(img, landmarker)

            if metrics.detection_ok:
                all_data.append({
                    "Category": cat,
                    "Mouth_Diff": metrics.mouth_corner_diff_norm,
                    "Total_Score": metrics.asymmetry_score
                })

    landmarker.close()
    df = pd.DataFrame(all_data)

    stats = df.groupby("Category")["Mouth_Diff"].agg(['mean', 'std', 'min', 'max']).reset_index()
    print("\n=== REZULTATY KALIBRACJI (DO RAPORTU) ===")
    print(stats)

    stats.to_csv("kalibracja_progow.csv", index=False)
    print("\nDane zapisane w kalibracja_progow.csv")


if __name__ == "__main__":
    run_calibration()