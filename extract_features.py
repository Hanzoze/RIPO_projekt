import cv2
import csv
import math
import xml.etree.ElementTree as ET
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm  # Рекомендую: pip install tqdm

# ============================================================
# KONFIGURACJA
# ============================================================
BASE_DIR = Path(r"C:\Users\danil\PycharmProjects\RIPO")
MODEL_PATH = str(BASE_DIR / "face_landmarker.task")

IMAGE_ROOTS = [
    BASE_DIR / "Image" / "Image",
    BASE_DIR / "Image2" / "Image2",
    BASE_DIR / "Image3" / "Image3",
    BASE_DIR / "Image4" / "Image4",
]
XML_ROOT = BASE_DIR / "Image_large_XML" / "Image_large_XML"
OUTPUT_DIR = BASE_DIR / "exploration_results"
OUTPUT_CSV = OUTPUT_DIR / "features.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Индексы MediaPipe
LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
LEFT_BROW_INNER, LEFT_BROW_OUTER = 107, 46
RIGHT_BROW_INNER, RIGHT_BROW_OUTER = 336, 276
NOSE_TIP, NOSE_BASE = 4, 2


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def midpoint(p1, p2):
    return (p1 + p2) / 2.0


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

        mouth_left_pt = get_pt(MOUTH_LEFT)
        mouth_right_pt = get_pt(MOUTH_RIGHT)
        midline_x = (get_pt(NOSE_TIP)[0] + get_pt(NOSE_BASE)[0]) / 2.0

        left_brow_h = (left_eye_c[1] - get_pt(LEFT_BROW_INNER)[1]) / ipd
        right_brow_h = (right_eye_c[1] - get_pt(RIGHT_BROW_INNER)[1]) / ipd

        # Возвращаем словарь со всеми вычисленными признаками
        return {
            "ipd_px": round(ipd, 3),
            "left_eye_open": round(left_eye_open, 4),
            "right_eye_open": round(right_eye_open, 4),
            "eye_aperture_asymmetry": round(abs(left_eye_open - right_eye_open), 4),
            "eye_aperture_ratio": round(min(left_eye_open, right_eye_open) / max(left_eye_open, right_eye_open + 1e-9),
                                        4),
            "mouth_corner_height_diff": round(abs(mouth_left_pt[1] - mouth_right_pt[1]) / ipd, 4),
            "mouth_left_deviation": round(abs(mouth_left_pt[0] - midline_x) / ipd, 4),
            "mouth_right_deviation": round(abs(mouth_right_pt[0] - midline_x) / ipd, 4),
            "mouth_deviation_asymmetry": round(
                abs(abs(mouth_left_pt[0] - midline_x) - abs(mouth_right_pt[0] - midline_x)) / ipd, 4),
            "left_brow_height": round(left_brow_h, 4),
            "right_brow_height": round(right_brow_h, 4),
            "brow_height_asymmetry": round(abs(left_brow_h - right_brow_h), 4),
            "brow_height_ratio": round(min(left_brow_h, right_brow_h) / max(abs(left_brow_h), abs(right_brow_h) + 1e-9),
                                       4),
            "mouth_width": round(distance(mouth_left_pt, mouth_right_pt) / ipd, 4),
            "mouth_angle": round(abs(math.degrees(
                math.atan2(mouth_right_pt[1] - mouth_left_pt[1], mouth_right_pt[0] - mouth_left_pt[0]))), 4),
            "eye_angle": round(
                abs(math.degrees(math.atan2(right_eye_c[1] - left_eye_c[1], right_eye_c[0] - left_eye_c[0]))), 4),
            "left_eye_width": round(distance(get_pt(LEFT_EYE_LEFT), get_pt(LEFT_EYE_RIGHT)) / ipd, 4),
            "right_eye_width": round(distance(get_pt(RIGHT_EYE_LEFT), get_pt(RIGHT_EYE_RIGHT)) / ipd, 4),
            "eye_width_asymmetry": round(abs((distance(get_pt(LEFT_EYE_LEFT), get_pt(LEFT_EYE_RIGHT)) - distance(
                get_pt(RIGHT_EYE_LEFT), get_pt(RIGHT_EYE_RIGHT))) / ipd), 4),
        }
    except:
        return None


# ============================================================
# SKANOWANIE I PARSOWANIE
# ============================================================
print("Skanowanie i mapowanie plików...")
all_images = {}
for img_root in IMAGE_ROOTS:
    if not img_root.exists(): continue
    for f in img_root.rglob("*"):
        if f.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}:
            # Klucz: "podfolder/nazwa" (np. "1/1")
            rel_key = str(f.relative_to(img_root).with_suffix('')).replace("\\", "/")
            all_images[rel_key] = f

labeled_samples = []
for xml_f in XML_ROOT.rglob("*.xml"):
    rel_key = str(xml_f.relative_to(XML_ROOT).with_suffix('')).replace("\\", "/")

    if rel_key in all_images:
        try:
            tree = ET.parse(xml_f)
            root = tree.getroot()
            names = [o.findtext("name", "") for o in root.findall("object")]
            label = "StrongPalsy" if any("Strong" in n for n in names) else \
                "SlightPalsy" if any("Slight" in n for n in names) else \
                    "Normal" if any("Normal" in n for n in names) else None

            if label:
                labeled_samples.append((all_images[rel_key], label, rel_key))
        except:
            continue

print(f"Znaleziono unikalnych par do przetworzenia: {len(labeled_samples)}")

# ============================================================
# PRZETWARZANIE MEDIA PIPE
# ============================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=False, num_faces=1)

results_rows = []
FEATURE_NAMES = ["ipd_px", "left_eye_open", "right_eye_open", "eye_aperture_asymmetry", "eye_aperture_ratio",
                 "mouth_corner_height_diff", "mouth_left_deviation", "mouth_right_deviation",
                 "mouth_deviation_asymmetry", "left_brow_height", "right_brow_height", "brow_height_asymmetry",
                 "brow_height_ratio", "mouth_width", "mouth_angle", "eye_angle", "left_eye_width", "right_eye_width",
                 "eye_width_asymmetry"]

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    # tqdm automatycznie pokaże pasek postępu i czas do końca
    for img_path, label, rel_key in tqdm(labeled_samples, desc="Analiza twarzy"):
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

# ============================================================
# ZAPIS
# ============================================================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "label"] + FEATURE_NAMES)
    writer.writeheader()
    writer.writerows(results_rows)

print(f"\nGotowe! Zapisano {len(results_rows)} wierszy.")
counts = Counter(r["label"] for r in results_rows)
for k, v in counts.items():
    print(f"  {k}: {v} ({v / len(results_rows) * 100:.1f}%)")