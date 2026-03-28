"""
Ocena symetrii twarzy z użyciem MediaPipe Face Mesh
Etap 2 - Prototyp Badawczy R&D
Autorzy: Danylchenko Illia (282633), Kijek Klaudyna (280891)

Kompatybilność: MediaPipe >= 0.10  (nowe API Tasks)
Instalacja:     pip install mediapipe opencv-python numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

# ──────────────────────────────────────────────────────────────
# Nowe API MediaPipe 0.10+ używa mediapipe.tasks
# ──────────────────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe import Image as MpImage, ImageFormat

# ──────────────────────────────────────────────────────────────
# Indeksy punktów MediaPipe Face Mesh (478 punktów w nowym API)
# Źródło: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
# ──────────────────────────────────────────────────────────────
LANDMARK_INDICES = {
    # Oczy – górna/dolna powieka
    "left_eye_top":      159,
    "left_eye_bottom":   145,
    "left_eye_inner":    133,
    "left_eye_outer":     33,
    "right_eye_top":     386,
    "right_eye_bottom":  374,
    "right_eye_inner":   362,
    "right_eye_outer":   263,
    # Brwi
    "left_brow_top":     105,
    "left_brow_inner":   107,
    "left_brow_outer":    46,
    "right_brow_top":    334,
    "right_brow_inner":  336,
    "right_brow_outer":  276,
    # Usta
    "mouth_left":         61,
    "mouth_right":       291,
    "mouth_top":          13,
    "mouth_bottom":       14,
    # Oś środkowa
    "nose_tip":            4,
    "nose_bridge":         6,
    "forehead":           10,
    "chin":              152,
    # Źrenice (refine landmarks – indeksy 468-477)
    "left_iris_center":  468,
    "right_iris_center": 473,
}


@dataclass
class AsymmetryMetrics:
    """Wszystkie mierzone wskaźniki asymetrii dla jednej klatki/zdjęcia."""
    ipd_px:                  float = 0.0
    eye_height_diff_norm:    float = 0.0
    eye_corner_height_diff:  float = 0.0
    brow_height_diff_norm:   float = 0.0
    mouth_corner_diff_norm:  float = 0.0
    mouth_width_asymmetry:   float = 0.0
    eye_closure_left:        float = 0.0
    eye_closure_right:       float = 0.0
    asymmetry_score:         float = 0.0
    hb_grade:                int   = 1
    hb_label:                str   = "Normal"
    detection_ok:            bool  = True
    error_msg:               str   = ""


def _get_model_path() -> str:
    """
    Zwraca ścieżkę do modelu face_landmarker.task.
    Pobiera automatycznie jeśli nie istnieje.
    """
    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if not os.path.exists(model_path):
        print("[INFO] Pobieranie modelu face_landmarker.task (~30 MB)...")
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print(f"[OK]   Model zapisany: {model_path}")
    return model_path


def create_face_landmarker(running_mode=RunningMode.IMAGE) -> FaceLandmarker:
    """Tworzy instancję FaceLandmarker (nowe API MediaPipe 0.10+)."""
    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_get_model_path()),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def _lm_to_px(landmark, w: int, h: int) -> np.ndarray:
    """Konwertuje znormalizowany landmark (0-1) na piksele."""
    return np.array([landmark.x * w, landmark.y * h], dtype=float)


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def assign_hb_grade(score: float) -> tuple[int, str]:
    """Progi empiryczne (kalibracja na zbiorze pilotażowym)."""
    thresholds = [
        (0.03, 1, "Normal (I)"),
        (0.07, 2, "Mild dysfunction (II)"),
        (0.13, 3, "Moderate dysfunction (III)"),
        (0.20, 4, "Moderately severe (IV)"),
        (0.30, 5, "Severe dysfunction (V)"),
        (1.00, 6, "Total paralysis (VI)"),
    ]
    for thresh, grade, label in thresholds:
        if score <= thresh:
            return grade, label
    return 6, "Total paralysis (VI)"


def analyze_frame(frame: np.ndarray, landmarker: FaceLandmarker) -> AsymmetryMetrics:
    """
    Przetwarza pojedynczą klatkę BGR i zwraca metryki asymetrii.
    Działa z nowym API MediaPipe 0.10+.
    """
    m = AsymmetryMetrics()
    h, w = frame.shape[:2]

    # MediaPipe 0.10+: konwersja BGR → RGB → MpImage
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        m.detection_ok = False
        m.error_msg = "No face detected"
        return m

    lm = result.face_landmarks[0]  # lista NormalizedLandmark

    # Pomocnik: pobierz punkt po indeksie
    def pt(key: str) -> np.ndarray:
        idx = LANDMARK_INDICES[key]
        if idx >= len(lm):
            # Fallback dla punktów tęczówki gdy brak refinement
            return np.array([0.0, 0.0])
        return _lm_to_px(lm[idx], w, h)

    # ── IPD ──────────────────────────────────────────────────
    # Używamy kącików oczu jako stabilne przybliżenie IPD
    left_inner  = pt("left_eye_inner")
    right_inner = pt("right_eye_inner")
    ipd = _dist(left_inner, right_inner)
    if ipd < 1.0:
        m.detection_ok = False
        m.error_msg = "IPD too small (face too far or partial detection)"
        return m
    m.ipd_px = round(ipd, 1)

    # ── Szpary powiekowe ─────────────────────────────────────
    eye_h_l = _dist(pt("left_eye_top"),  pt("left_eye_bottom"))  / ipd
    eye_h_r = _dist(pt("right_eye_top"), pt("right_eye_bottom")) / ipd
    m.eye_closure_left  = round(eye_h_l, 4)
    m.eye_closure_right = round(eye_h_r, 4)
    m.eye_height_diff_norm = round(abs(eye_h_l - eye_h_r), 4)

    # ── Kąciki oczu (Y) ──────────────────────────────────────
    m.eye_corner_height_diff = round(
        abs(pt("left_eye_outer")[1] - pt("right_eye_outer")[1]) / ipd, 4
    )

    # ── Brwi ─────────────────────────────────────────────────
    nose_y = pt("nose_bridge")[1]
    brow_l = abs(nose_y - pt("left_brow_top")[1])  / ipd
    brow_r = abs(nose_y - pt("right_brow_top")[1]) / ipd
    m.brow_height_diff_norm = round(abs(brow_l - brow_r), 4)

    # ── Kąciki ust ───────────────────────────────────────────
    ml = pt("mouth_left")
    mr = pt("mouth_right")
    mt = pt("mouth_top")
    m.mouth_corner_diff_norm = round(abs(ml[1] - mr[1]) / ipd, 4)
    m.mouth_width_asymmetry  = round(
        abs(abs(mt[0] - ml[0]) - abs(mt[0] - mr[0])) / ipd, 4
    )

    # ── Asymmetry Score (ważona suma) ────────────────────────
    score = (
        0.25 * m.eye_height_diff_norm    +
        0.15 * m.eye_corner_height_diff  +
        0.20 * m.brow_height_diff_norm   +
        0.25 * m.mouth_corner_diff_norm  +
        0.15 * m.mouth_width_asymmetry
    )
    m.asymmetry_score = round(score, 4)
    m.hb_grade, m.hb_label = assign_hb_grade(score)
    return m


def draw_overlay(frame: np.ndarray, landmarker: FaceLandmarker,
                 metrics: AsymmetryMetrics) -> np.ndarray:
    """Rysuje punkty kluczowe i panel z wynikami na klatce."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    if not metrics.detection_ok:
        cv2.putText(vis, f"NO FACE: {metrics.error_msg}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

    # Pobierz landmarks jeszcze raz do rysowania
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        # Rysuj wybrane punkty kluczowe
        for key, idx in LANDMARK_INDICES.items():
            if idx >= len(lm):
                continue
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            color = (0, 200, 255) if "mouth" in key else \
                    (255, 100, 0) if "brow"  in key else \
                    (0, 255, 100)
            cv2.circle(vis, (x, y), 3, color, -1)

        # Oś środkowa twarzy
        p_fore = (int(lm[10].x*w),  int(lm[10].y*h))
        p_chin = (int(lm[152].x*w), int(lm[152].y*h))
        cv2.line(vis, p_fore, p_chin, (255, 220, 50), 1)

    # ── Panel informacyjny ────────────────────────────────────
    def grade_color(g):
        if g <= 1: return (0, 220, 0)
        if g <= 3: return (0, 165, 255)
        return (0, 0, 255)

    panel_lines = [
        (f"HB Grade: {metrics.hb_grade}  {metrics.hb_label}",
         grade_color(metrics.hb_grade), 0.9),
        (f"Asym Score:     {metrics.asymmetry_score:.4f}", (210, 210, 210), 0.62),
        (f"Eye height diff:{metrics.eye_height_diff_norm:.4f}", (210, 210, 210), 0.58),
        (f"Brow diff:      {metrics.brow_height_diff_norm:.4f}", (210, 210, 210), 0.58),
        (f"Mouth corner:   {metrics.mouth_corner_diff_norm:.4f}", (210, 210, 210), 0.58),
        (f"Eye open L/R:   {metrics.eye_closure_left:.3f} / {metrics.eye_closure_right:.3f}",
         (210, 210, 210), 0.58),
        (f"IPD: {metrics.ipd_px:.1f} px", (150, 150, 150), 0.52),
    ]

    # Półprzezroczyste tło panelu
    overlay = vis.copy()
    cv2.rectangle(overlay, (5, 5), (400, 210), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    y0 = 28
    for text, color, scale in panel_lines:
        cv2.putText(vis, text, (12, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y0 += int(scale * 36) + 3

    return vis


def run_batch(input_dir: str):
    """Przetwarza wsadowo wszystkie zdjęcia z katalogu."""
    out_dir = os.path.join(input_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    landmarker = create_face_landmarker(RunningMode.IMAGE)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted(f for f in os.listdir(input_dir)
                   if os.path.splitext(f.lower())[1] in exts)

    if not files:
        print(f"[WARN] Brak zdjęć w: {input_dir}")
        landmarker.close()
        return

    print(f"\n=== Przetwarzanie {len(files)} zdjęć ===")
    all_results = []

    for fname in files:
        frame = cv2.imread(os.path.join(input_dir, fname))
        if frame is None:
            print(f"  [ERR] Nie można wczytać: {fname}")
            continue

        m = analyze_frame(frame, landmarker)
        vis = draw_overlay(frame, landmarker, m)
        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_result.jpg"), vis)

        status = "OK" if m.detection_ok else f"FAIL ({m.error_msg})"
        print(f"  [{status}] {base}: HB={m.hb_grade}, score={m.asymmetry_score:.4f}")

        d = asdict(m)
        d["filename"] = fname
        all_results.append(d)

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Wyniki: {json_path}")
    print(f"[DONE] Wizualizacje: {out_dir}/")
    landmarker.close()


def run_camera():
    """Tryb kamerowy – analiza na żywo (MediaPipe 0.10+)."""
    # W trybie VIDEO landmarker działa szybciej niż wywołanie IMAGE per klatka
    landmarker = create_face_landmarker(RunningMode.IMAGE)

    # Windows: użyj CAP_DSHOW dla pewniejszego otwarcia kamery
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)   # fallback bez DSHOW
    if not cap.isOpened():
        print("[ERR] Nie można otworzyć kamery.")
        landmarker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WIN_NAME = "Face Symmetry Analysis – Etap 2  [q=quit, s=save]"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)   # jawne stworzenie okna
    cv2.resizeWindow(WIN_NAME, 800, 600)

    print("Kamera uruchomiona. Naciśnij 'q' = wyjście, 's' = zrzut ekranu.")
    print("Jeśli okno nie jest widoczne – sprawdź pasek zadań Windows.")
    snap_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Brak klatki z kamery.")
            break

        metrics = analyze_frame(frame, landmarker)
        vis     = draw_overlay(frame, landmarker, metrics)

        cv2.imshow(WIN_NAME, vis)
        cv2.waitKey(1)   # Windows wymaga tego żeby okno się odświeżyło

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"snapshot_{snap_idx:03d}.jpg"
            cv2.imwrite(fname, vis)
            print(f"  [SAVED] {fname}")
            snap_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "camera":
        run_camera()
    elif sys.argv[1] == "batch" and len(sys.argv) >= 3:
        run_batch(sys.argv[2])
    else:
        print("Użycie:")
        print("  python analyze.py camera          # kamera na żywo")
        print("  python analyze.py batch <katalog> # wsadowe przetwarzanie zdjęć")