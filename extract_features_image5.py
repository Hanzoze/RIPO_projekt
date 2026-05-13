import cv2
import csv
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Konfiguracja ścieżek
BASE_DIR = Path(r"C:\Users\danil\PycharmProjects\RIPO")
IMAGE5_DIR = BASE_DIR / "Image5"
MODEL_PATH = str(BASE_DIR / "face_landmarker.task")
OUTPUT_CSV = BASE_DIR / "exploration_results" / "features_image5.csv"

# Indeksy punktów MediaPipe (zgodnie z Twoim oryginalnym skryptem)
LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LEFT_BROW_INNER = 107
RIGHT_BROW_INNER = 336
NOSE_TIP, NOSE_BASE = 4, 2

def distance(p1, p2): return np.linalg.norm(p1 - p2)
def midpoint(p1, p2): return (p1 + p2) / 2.0

def extract_features(landmarks, w, h):
    try:
        def get_pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        left_eye_c = midpoint(get_pt(LEFT_EYE_LEFT), get_pt(LEFT_EYE_RIGHT))
        right_eye_c = midpoint(get_pt(RIGHT_EYE_LEFT), get_pt(RIGHT_EYE_RIGHT))
        ipd = distance(left_eye_c, right_eye_c)
        if ipd < 1e-6: return None

        left_eye_open = distance(get_pt(LEFT_EYE_TOP), get_pt(LEFT_EYE_BOTTOM)) / ipd
        right_eye_open = distance(get_pt(RIGHT_EYE_TOP), get_pt(RIGHT_EYE_BOTTOM)) / ipd
        mouth_left_pt, mouth_right_pt = get_pt(MOUTH_LEFT), get_pt(MOUTH_RIGHT)
        midline_x = (get_pt(NOSE_TIP)[0] + get_pt(NOSE_BASE)[0]) / 2.0
        left_brow_h = (left_eye_c[1] - get_pt(LEFT_BROW_INNER)[1]) / ipd
        right_brow_h = (right_eye_c[1] - get_pt(RIGHT_BROW_INNER)[1]) / ipd

        return {
            "ipd_px": round(ipd, 3),
            "left_eye_open": round(left_eye_open, 4),
            "right_eye_open": round(right_eye_open, 4),
            "eye_aperture_asymmetry": round(abs(left_eye_open - right_eye_open), 4),
            "eye_aperture_ratio": round(min(left_eye_open, right_eye_open) / max(left_eye_open, right_eye_open + 1e-9), 4),
            "mouth_corner_height_diff": round(abs(mouth_left_pt[1] - mouth_right_pt[1]) / ipd, 4),
            "mouth_left_deviation": round(abs(mouth_left_pt[0] - midline_x) / ipd, 4),
            "mouth_right_deviation": round(abs(mouth_right_pt[0] - midline_x) / ipd, 4),
            "mouth_deviation_asymmetry": round(abs(abs(mouth_left_pt[0] - midline_x) - abs(mouth_right_pt[0] - midline_x)) / ipd, 4),
            "left_brow_height": round(left_brow_h, 4),
            "right_brow_height": round(right_brow_h, 4),
            "brow_height_asymmetry": round(abs(left_brow_h - right_brow_h), 4),
            "brow_height_ratio": round(min(left_brow_h, right_brow_h) / max(abs(left_brow_h), abs(right_brow_h) + 1e-9), 4),
            "mouth_width": round(distance(mouth_left_pt, mouth_right_pt) / ipd, 4),
            "mouth_angle": round(abs(math.degrees(math.atan2(mouth_right_pt[1] - mouth_left_pt[1], mouth_right_pt[0] - mouth_left_pt[0]))), 4),
            "eye_angle": round(abs(math.degrees(math.atan2(right_eye_c[1] - left_eye_c[1], right_eye_c[0] - left_eye_c[0]))), 4),
            "left_eye_width": round(distance(get_pt(LEFT_EYE_LEFT), get_pt(LEFT_EYE_RIGHT)) / ipd, 4),
            "right_eye_width": round(distance(get_pt(RIGHT_EYE_LEFT), get_pt(RIGHT_EYE_RIGHT)) / ipd, 4),
            "eye_width_asymmetry": round(abs((distance(get_pt(LEFT_EYE_LEFT), get_pt(LEFT_EYE_RIGHT)) - distance(get_pt(RIGHT_EYE_LEFT), get_pt(RIGHT_EYE_RIGHT))) / ipd), 4),
        }
    except: return None

# --- POPRAWIONA LOGIKA ZBIERANIA ZDJĘĆ ---
image_tasks = []

# Przeszukujemy wszystkie podfoldery w Image5 (np. "33", "34", "35")
if IMAGE5_DIR.exists():
    for patient_dir in IMAGE5_DIR.iterdir():
        if patient_dir.is_dir(): # Poprawione z is_all_dir() na is_dir()
            # Pobieramy ID pacjenta z nazwy folderu (np. "33")
            patient_id = patient_dir.name.split(":")[0]

            # rglob("*") znajdzie pliki we wszystkich podfolderach, np. w "strong"
            for img_f in patient_dir.rglob("*"):
                if img_f.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}:
                    # Przypisujemy etykietę StrongPalsy
                    # rel_path będzie użyte jako video_id w treningu (np. Image5_33)
                    rel_path = f"Image5_{patient_id}/{img_f.name}"
                    image_tasks.append((img_f, "StrongPalsy", rel_path))
else:
    print(f"[ERROR] Folder {IMAGE5_DIR} nie istnieje!")

print(f"Znaleziono {len(image_tasks)} zdjęć w Image5.")

# Przetwarzanie
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
results_rows = []

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    for img_path, label, rel_key in tqdm(image_tasks, desc="Analiza Image5"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        h, w = img_bgr.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        detection_result = landmarker.detect(mp_image)
        if detection_result.face_landmarks:
            features = extract_features(detection_result.face_landmarks[0], w, h)
            if features:
                row = {"filename": rel_key, "label": label}
                row.update(features)
                results_rows.append(row)

# Zapis (identyczny zestaw pól jak w Twoim features.csv)
cols = ["filename", "label", "ipd_px", "left_eye_open", "right_eye_open", "eye_aperture_asymmetry",
        "eye_aperture_ratio", "mouth_corner_height_diff", "mouth_left_deviation", "mouth_right_deviation",
        "mouth_deviation_asymmetry", "left_brow_height", "right_brow_height", "brow_height_asymmetry",
        "brow_height_ratio", "mouth_width", "mouth_angle", "eye_angle", "left_eye_width", "right_eye_width",
        "eye_width_asymmetry"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(results_rows)

print(f"Gotowe! Zapisano {len(results_rows)} wierszy z Image5.")