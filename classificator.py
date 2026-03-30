import cv2
import math
import argparse
import numpy as np
import mediapipe as mp
import joblib
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# KONFIGURACJA
# ============================================================
BASE_DIR   = Path(r"C:\Users\danil\PycharmProjects\RIPO")
MODEL_PATH = BASE_DIR / "exploration_results" / "best_model.pkl"
OUTPUT_DIR = BASE_DIR / "exploration_results"

IMAGE_ROOTS = [
    BASE_DIR / "Image"  / "Image",
    BASE_DIR / "Image2" / "Image2",
    BASE_DIR / "Image3" / "Image3",
    BASE_DIR / "Image4" / "Image4",
]
XML_ROOT = BASE_DIR / "Image_large_XML" / "Image_large_XML"

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}

FEATURE_COLS = [
    "left_eye_open", "right_eye_open",
    "eye_aperture_asymmetry", "eye_aperture_ratio",
    "mouth_corner_height_diff",
    "mouth_left_deviation", "mouth_right_deviation", "mouth_deviation_asymmetry",
    "left_brow_height", "right_brow_height",
    "brow_height_asymmetry", "brow_height_ratio",
    "mouth_width", "mouth_angle", "eye_angle",
    "left_eye_width", "right_eye_width", "eye_width_asymmetry",
]

LABEL_COLORS = {
    "Normal":      "#2ecc71",
    "SlightPalsy": "#e67e22",
    "StrongPalsy": "#e74c3c",
}
LABEL_PL = {
    "Normal":      "Brak porażenia",
    "SlightPalsy": "Lekkie porażenie",
    "StrongPalsy": "Silne porażenie",
}

# Indeksy MediaPipe
LEFT_EYE_LEFT, LEFT_EYE_RIGHT   = 33, 133
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
LEFT_EYE_TOP,   LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP,  RIGHT_EYE_BOTTOM = 386, 374
MOUTH_LEFT,  MOUTH_RIGHT        = 61, 291
MOUTH_TOP,   MOUTH_BOTTOM       = 13, 14
LEFT_BROW_INNER, RIGHT_BROW_INNER = 107, 336
NOSE_TIP, NOSE_BASE             = 4, 2

# ============================================================
# FUNKCJE
# ============================================================

def get_pt(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h])

def dist(a, b):
    return np.linalg.norm(a - b)

def mid(a, b):
    return (a + b) / 2.0

def extract_features(landmarks, w, h):
    lec = mid(get_pt(landmarks, LEFT_EYE_LEFT, w, h),  get_pt(landmarks, LEFT_EYE_RIGHT, w, h))
    rec = mid(get_pt(landmarks, RIGHT_EYE_LEFT, w, h), get_pt(landmarks, RIGHT_EYE_RIGHT, w, h))
    ipd = dist(lec, rec)
    if ipd < 1e-6:
        return None, None

    leo = dist(get_pt(landmarks, LEFT_EYE_TOP,  w, h), get_pt(landmarks, LEFT_EYE_BOTTOM, w, h)) / ipd
    reo = dist(get_pt(landmarks, RIGHT_EYE_TOP, w, h), get_pt(landmarks, RIGHT_EYE_BOTTOM, w, h)) / ipd
    eye_asym  = abs(leo - reo)
    eye_ratio = min(leo, reo) / max(leo, reo + 1e-9)

    ml = get_pt(landmarks, MOUTH_LEFT,  w, h)
    mr = get_pt(landmarks, MOUTH_RIGHT, w, h)
    mch_diff = abs(ml[1] - mr[1]) / ipd

    nose_x = (get_pt(landmarks, NOSE_TIP, w, h)[0] + get_pt(landmarks, NOSE_BASE, w, h)[0]) / 2
    mld = abs(ml[0] - nose_x) / ipd
    mrd = abs(mr[0] - nose_x) / ipd
    mdev_asym = abs(mld - mrd)

    lbh = (lec[1] - get_pt(landmarks, LEFT_BROW_INNER,  w, h)[1]) / ipd
    rbh = (rec[1] - get_pt(landmarks, RIGHT_BROW_INNER, w, h)[1]) / ipd
    brow_asym  = abs(lbh - rbh)
    brow_ratio = min(lbh, rbh) / max(abs(lbh), abs(rbh) + 1e-9)

    mw = dist(ml, mr) / ipd
    dx_m = mr[0] - ml[0]; dy_m = mr[1] - ml[1]
    m_angle = abs(math.degrees(math.atan2(dy_m, dx_m)))
    dx_e = rec[0] - lec[0]; dy_e = rec[1] - lec[1]
    e_angle = abs(math.degrees(math.atan2(dy_e, dx_e)))

    lew = dist(get_pt(landmarks, LEFT_EYE_LEFT,  w, h), get_pt(landmarks, LEFT_EYE_RIGHT, w, h)) / ipd
    rew = dist(get_pt(landmarks, RIGHT_EYE_LEFT, w, h), get_pt(landmarks, RIGHT_EYE_RIGHT, w, h)) / ipd
    ew_asym = abs(lew - rew)

    features = [leo, reo, eye_asym, eye_ratio, mch_diff, mld, mrd, mdev_asym,
                lbh, rbh, brow_asym, brow_ratio, mw, m_angle, e_angle, lew, rew, ew_asym]

    named = dict(zip(FEATURE_COLS, features))
    named["ipd_px"] = round(ipd, 2)
    named["lec"] = lec; named["rec"] = rec
    named["ml"] = ml;   named["mr"] = mr
    named["leo"] = leo; named["reo"] = reo
    named["lbh"] = lbh; named["rbh"] = rbh

    return features, named

def get_true_label(img_stem):
    """Szuka etykiety dla zdjęcia w XMLach."""
    for subdir in XML_ROOT.iterdir():
        if not subdir.is_dir(): continue
        xml_f = subdir / f"{img_stem}.xml"
        if xml_f.exists():
            try:
                root = ET.parse(xml_f).getroot()
                names = [o.findtext("name","") for o in root.findall("object")]
                if any("Strong" in n for n in names): return "StrongPalsy"
                if any("Slight" in n for n in names): return "SlightPalsy"
                if any("Normal" in n for n in names): return "Normal"
            except: pass
    return None

def collect_all_images():
    imgs = {}
    for root in IMAGE_ROOTS:
        if not root.exists(): continue
        for sub in root.iterdir():
            if sub.is_dir():
                for f in sub.iterdir():
                    if f.suffix.lower() in IMAGE_EXTENSIONS:
                        imgs[f.stem] = f
    return imgs

# ============================================================
# ARGUMENT PARSER
# ============================================================
parser = argparse.ArgumentParser(description="Demo klasyfikatora porażenia twarzy")
parser.add_argument("--image",  type=str, default=None, help="Ścieżka do zdjęcia")
parser.add_argument("--random", action="store_true",    help="Losowe zdjęcie z datasetu")
parser.add_argument("--all_classes", action="store_true",
                    help="Pokaż po jednym przykładzie z każdej klasy")
args = parser.parse_args()

# ============================================================
# WCZYTANIE MODELU
# ============================================================
print("Wczytywanie modelu...")
model = joblib.load(MODEL_PATH)
print(f"  Model załadowany: {MODEL_PATH.name}")

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH.parent.parent / "face_landmarker.task"))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    num_faces=1
)

# Tworzymy instancję landmarker'a
landmarker = vision.FaceLandmarker.create_from_options(options)

# ============================================================
# WYBÓR ZDJĘĆ DO TESTU
# ============================================================
all_images = collect_all_images()

if args.all_classes:
    # Znajdź po 1 przykładzie z każdej klasy
    samples = {}
    for stem, path in all_images.items():
        lbl = get_true_label(stem)
        if lbl and lbl not in samples:
            samples[lbl] = (stem, path)
        if len(samples) == 3:
            break
    test_images = [(p, s, get_true_label(s)) for s, p in samples.values()]
elif args.image:
    p = Path(args.image)
    test_images = [(p, p.stem, get_true_label(p.stem))]
else:
    # Domyślnie: losowe zdjęcie
    stem, path = random.choice(list(all_images.items()))
    test_images = [(path, stem, get_true_label(stem))]

# ============================================================
# PRZETWARZANIE I WIZUALIZACJA
# ============================================================

def process_and_visualize(img_path, stem, true_label):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Nie można wczytać: {img_path}")
        return

    # Konwersja kolorów i formatu dla MediaPipe
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # DETEKCJA (Tasks API)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        print(f"Brak wykrytej twarzy: {stem}")
        return

    # Wyciągamy pierwszą twarz (lista landmarków)
    landmarks = result.face_landmarks[0]

    # Ekstrakcja cech (funkcja extract_features pozostaje bez zmian)
    feat_vec, feat_named = extract_features(landmarks, w, h)
    if feat_vec is None:
        print(f"Błąd ekstrakcji cech: {stem}")
        return

    # Predykcja modelem ML
    pred_label = model.predict([feat_vec])[0]
    pred_proba = model.predict_proba([feat_vec])[0]
    classes = model.classes_

    # ---- Rysowanie na zdjęciu ----
    img_draw = img_rgb.copy()

    def to_int(pt):
        return (int(pt[0]), int(pt[1]))

    ipd = feat_named["ipd_px"]
    lec = feat_named["lec"]; rec = feat_named["rec"]
    ml  = feat_named["ml"];  mr  = feat_named["mr"]

    # Linia między oczami (IPD)
    cv2.line(img_draw, to_int(lec), to_int(rec), (100, 200, 255), 2)
    cv2.circle(img_draw, to_int(lec), 5, (100, 200, 255), -1)
    cv2.circle(img_draw, to_int(rec), 5, (100, 200, 255), -1)

    # Kąciki ust
    cv2.circle(img_draw, to_int(ml), 6, (255, 165, 0), -1)
    cv2.circle(img_draw, to_int(mr), 6, (255, 165, 0), -1)
    cv2.line(img_draw, to_int(ml), to_int(mr), (255, 165, 0), 2)

    # Pionowa linia środkowa
    nose_x = int((get_pt(landmarks, NOSE_TIP, w, h)[0] + get_pt(landmarks, NOSE_BASE, w, h)[0]) / 2)
    cv2.line(img_draw, (nose_x, 0), (nose_x, h), (200, 200, 200), 1)

    # Punkty powiek
    for idx, color in [(LEFT_EYE_TOP, (0,255,0)), (LEFT_EYE_BOTTOM, (0,255,0)),
                       (RIGHT_EYE_TOP, (0,200,0)), (RIGHT_EYE_BOTTOM, (0,200,0))]:
        cv2.circle(img_draw, to_int(get_pt(landmarks, idx, w, h)), 4, color, -1)

    # ---- Figura ----
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("#1a1a2e")

    # Panel 1: zdjęcie z adnotacjami
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_draw)
    ax1.set_title(f"Zdjęcie: {stem}", color="white", fontsize=10)
    ax1.axis("off")

    # Legenda do zdjęcia
    legend_items = [
        mpatches.Patch(color=(100/255, 200/255, 255/255), label=f"IPD = {ipd:.1f}px"),
        mpatches.Patch(color=(1, 165/255, 0), label="Linia ust"),
        mpatches.Patch(color=(0, 1, 0), label="Powieki"),
        mpatches.Patch(color=(0.8, 0.8, 0.8), label="Oś środkowa"),
    ]
    ax1.legend(handles=legend_items, loc="lower left",
               fontsize=7, facecolor="#2d2d44", labelcolor="white")

    # Panel 2: prawdopodobieństwa predykcji
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor("#2d2d44")
    bar_colors = [LABEL_COLORS.get(c, "#888") for c in classes]
    bars = ax2.barh(classes, pred_proba, color=bar_colors, edgecolor="white", height=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Prawdopodobieństwo", color="white")
    ax2.set_title("Wynik klasyfikatora", color="white", fontsize=11)
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("white")
    ax2.spines["left"].set_color("white")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for bar, prob in zip(bars, pred_proba):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{prob:.3f}", va="center", color="white", fontsize=10)

    # Etykieta predykowana
    pred_color = LABEL_COLORS.get(pred_label, "#888")
    ax2.set_title(
        f"Predykcja: {LABEL_PL.get(pred_label, pred_label)}\n"
        f"(p={max(pred_proba):.2f})",
        color=pred_color, fontsize=12, fontweight="bold"
    )

    # Prawdziwa etykieta
    if true_label:
        match = "✓ POPRAWNA" if pred_label == true_label else "✗ BŁĘDNA"
        true_color = "#2ecc71" if pred_label == true_label else "#e74c3c"
        ax2.text(0.5, -0.12,
                 f"Prawdziwa: {LABEL_PL.get(true_label, true_label)}  {match}",
                 transform=ax2.transAxes, ha="center",
                 color=true_color, fontsize=10, fontweight="bold")

    # Panel 3: wartości kluczowych cech
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor("#2d2d44")

    key_features = [
        ("eye_aperture_asymmetry",   "Asymetria powiek",     0.05),
        ("eye_aperture_ratio",       "Stosunek otwarcia oczu", 0.85),
        ("mouth_corner_height_diff", "Różnica kącików ust",  0.04),
        ("mouth_angle",              "Kąt ust (°)",          5.0),
        ("brow_height_asymmetry",    "Asymetria brwi",       0.03),
        ("eye_width_asymmetry",      "Asymetria szerokości oczu", 0.02),
    ]

    feat_names_short = [f[1] for f in key_features]
    feat_vals        = [feat_named.get(f[0], 0) for f in key_features]
    thresholds       = [f[2] for f in key_features]

    bar_c = ["#e74c3c" if v > t else "#2ecc71"
             for v, t in zip(feat_vals, thresholds)]
    bars3 = ax3.barh(feat_names_short, feat_vals, color=bar_c, edgecolor="white", height=0.5)
    ax3.set_title("Kluczowe cechy asymetrii\n(czerwony = przekroczony próg)",
                  color="white", fontsize=10)
    ax3.tick_params(colors="white", labelsize=8)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)
    ax3.spines["bottom"].set_color("white")
    ax3.spines["left"].set_color("white")

    # Progi jako pionowe linie
    for i, (_, _, thresh) in enumerate(key_features):
        ax3.plot([thresh, thresh], [i - 0.4, i + 0.4],
                 color="yellow", linewidth=1.5, linestyle="--")

    for bar, val in zip(bars3, feat_vals):
        ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", color="white", fontsize=8)

    plt.tight_layout(pad=2)
    fig.suptitle("Demo: Ocena symetrii twarzy", color="white",
                 fontsize=13, fontweight="bold", y=1.01)

    out_path = OUTPUT_DIR / f"demo_{stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\nWynik zapisany: {out_path}")

    # Podsumowanie w konsoli
    print(f"\n{'='*50}")
    print(f"WYNIK DLA: {stem}")
    print(f"{'='*50}")
    print(f"  Predykcja:       {pred_label} ({LABEL_PL[pred_label]})")
    if true_label:
        status = "✓ POPRAWNA" if pred_label == true_label else "✗ BŁĘDNA"
        print(f"  Prawdziwa klasa: {true_label} → {status}")
    print(f"  Prawdopodobieństwa:")
    for cls, prob in zip(classes, pred_proba):
        print(f"    {cls:15s}: {prob:.4f}")
    print(f"\n  Kluczowe cechy:")
    print(f"    IPD:                     {feat_named['ipd_px']:.1f} px")
    print(f"    Asymetria powiek:        {feat_named['eye_aperture_asymmetry']:.4f}")
    print(f"    Stosunek otwarcia oczu:  {feat_named['eye_aperture_ratio']:.4f}")
    print(f"    Różnica kącików ust:     {feat_named['mouth_corner_height_diff']:.4f}")
    print(f"    Asymetria brwi:          {feat_named['brow_height_asymmetry']:.4f}")

# Uruchomienie
for img_path, stem, true_label in test_images:
    process_and_visualize(img_path, stem, true_label)

landmarker.close()
print("\nDemo zakończone.")