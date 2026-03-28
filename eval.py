import os
import cv2
import pandas as pd
from analyze import analyze_frame, create_face_landmarker
from mediapipe.tasks.python.vision import RunningMode

# Ścieżki z Twojego zrzutu ekranu i YAML
IMG_DIR = r"Dataset/Dataset/images/valid"
LBL_DIR = r"Dataset/Dataset/labels/valid"


def run_evaluation():
    landmarker = create_face_landmarker(RunningMode.IMAGE)
    results = []

    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))]

    print(f"Rozpoczynam skanowanie {len(image_files)} zdjęć...")

    for img_name in image_files:
        img_path = os.path.join(IMG_DIR, img_name)
        lbl_path = os.path.join(LBL_DIR, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # 1. Odczyt obrazu i analiza MediaPipe
        frame = cv2.imread(img_path)
        metrics = analyze_frame(frame, landmarker)

        # 2. Odczyt etykiety (Ground Truth)
        has_drooping_label = False
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                content = f.read().strip()
                if content:
                    # Zakładamy, że klasa > 0 w Twoich plikach to jakaś forma asymetrii
                    has_drooping_label = True

                    # 3. Logika porównania (Twoja inżynierska intuicja)
        # Przyjmijmy próg asymetrii np. 0.05 jako "wykrycie problemu"
        detected_asymmetry = metrics.asymmetry_score > 0.05

        results.append({
            "file": img_name,
            "score": metrics.asymmetry_score,
            "label_exists": has_drooping_label,
            "correct": detected_asymmetry == has_drooping_label
        })

    # Generowanie statystyk
    df = pd.DataFrame(results)
    accuracy = df['correct'].mean()

    print("\n=== STATYSTYKI PROTOTYPU ===")
    print(f"Ogólna dokładność (Accuracy): {accuracy:.2%}")
    print(f"Średni wynik asymetrii: {df['score'].mean():.4f}")

    # Zapis do CSV dla raportu (Wymóg Etapu 2)
    df.to_csv("wyniki_eksperymentu.csv", index=False)
    print("Wyniki zapisano do wyniki_eksperymentu.csv")

    landmarker.close()


if __name__ == "__main__":
    run_evaluation()