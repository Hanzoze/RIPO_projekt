import cv2
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# --- KONFIGURACJA ŚCIEŻEK ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "exploration_results_v1_4" / "final_model_v1_4.pkl"
# Ścieżka do nowych danych (np. plik CSV z nowymi pacjentami)
VALIDATION_DATA_PATH = BASE_DIR / "exploration_results" / "features.csv"

# --- PARAMETRY ARCHITEKTURY v1.4 ---
LABEL_ORDER = ["Normal", "SlightPalsy", "StrongPalsy"]
TEMPORAL_WINDOW = 5
TEMPORAL_SOURCE_COLS = [
    "eye_aperture_asymmetry", "mouth_corner_height_diff", "brow_height_asymmetry",
    "mouth_deviation_asymmetry", "global_symmetry_score", "lower_face_severity", "eye_closure_min"
]

# Dokładna lista kolumn, na których trenowany był model v1.4
ALL_FEATURE_COLS = [
    "left_eye_open", "right_eye_open", "eye_aperture_asymmetry", "eye_aperture_ratio",
    "mouth_corner_height_diff", "mouth_left_deviation", "mouth_right_deviation",
    "mouth_deviation_asymmetry", "left_brow_height", "right_brow_height",
    "brow_height_asymmetry", "brow_height_ratio", "mouth_width", "mouth_angle",
    "eye_angle", "left_eye_width", "right_eye_width", "eye_width_asymmetry",
    "HB_composite_index", "eye_mouth_asymmetry_product", "global_symmetry_score",
    "brow_eye_combined", "lower_face_severity", "eye_closure_min",
    "mouth_asymmetry_combined", "total_asymmetry_delta",
    "delta_eye_aperture_asymmetry", "roll_mean_eye_aperture_asymmetry", "roll_std_eye_aperture_asymmetry",
    "delta_mouth_corner_height_diff", "roll_mean_mouth_corner_height_diff", "roll_std_mouth_corner_height_diff",
    "delta_brow_height_asymmetry", "roll_mean_brow_height_asymmetry", "roll_std_brow_height_asymmetry",
    "delta_mouth_deviation_asymmetry", "roll_mean_mouth_deviation_asymmetry", "roll_std_mouth_deviation_asymmetry",
    "delta_global_symmetry_score", "roll_mean_global_symmetry_score", "roll_std_global_symmetry_score",
    "delta_lower_face_severity", "roll_mean_lower_face_severity", "roll_std_lower_face_severity",
    "delta_eye_closure_min", "roll_mean_eye_closure_min", "roll_std_eye_closure_min"
]


def engineer_features_v1_4(df):
    """Odtworzenie cech statycznych z v1.4[cite: 27]."""
    d = df.copy()
    d["HB_composite_index"] = (d["eye_aperture_asymmetry"] * 3.0 + d["mouth_angle"] / 20.0 + d[
        "mouth_corner_height_diff"] * 2.0) / 6.0
    d["global_symmetry_score"] = np.sqrt(
        d["eye_aperture_asymmetry"] ** 2 + d["mouth_corner_height_diff"] ** 2 + d["brow_height_asymmetry"] ** 2 + d[
            "mouth_deviation_asymmetry"] ** 2)
    d["lower_face_severity"] = (d["mouth_corner_height_diff"] * 2.0 + d["mouth_angle"] / 15.0 + d[
        "mouth_deviation_asymmetry"]) / 4.0
    d["eye_closure_min"] = d[["left_eye_open", "right_eye_open"]].min(axis=1)
    # Pozostałe cechy pomocnicze użyte w treningu
    d["eye_mouth_asymmetry_product"] = d["eye_aperture_asymmetry"] * d["mouth_corner_height_diff"]
    d["brow_eye_combined"] = (d["brow_height_asymmetry"] + d["eye_aperture_asymmetry"]) / 2.0
    d["mouth_asymmetry_combined"] = np.sqrt(
        d["mouth_left_deviation"] ** 2 + d["mouth_right_deviation"] ** 2 + d["mouth_corner_height_diff"] ** 2)
    d["total_asymmetry_delta"] = d.select_dtypes(include=[np.number]).abs().sum(axis=1)
    return d


def add_temporal_features_validation(df):
    """Dodanie cech temporalnych per-pacjent[cite: 34, 50]."""
    groups = []
    for vid, grp in df.groupby("video_id", sort=False):
        grp = grp.copy()
        for col in TEMPORAL_SOURCE_COLS:
            series = grp[col]
            grp[f"delta_{col}"] = series.diff().fillna(0.0)
            grp[f"roll_mean_{col}"] = series.rolling(TEMPORAL_WINDOW, min_periods=1).mean()
            grp[f"roll_std_{col}"] = series.rolling(TEMPORAL_WINDOW, min_periods=1).std().fillna(0.0)
        groups.append(grp)
    return pd.concat(groups, ignore_index=True)


def apply_validation_normalization(df):
    """Normalizacja baseline per-pacjent dla zbioru walidacyjnego."""
    # Lista wszystkich kolumn (47), które model widział w treningu
    # Warto pobrać ją dynamicznie z danych treningowych, tutaj lista uproszczona
    feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "video_id" in feat_cols: feat_cols.remove("video_id")

    normalized = []
    for vid, group in df.groupby("video_id"):
        group = group.copy()
        # Szukamy klatek Normal u pacjenta jako punktu odniesienia
        baseline_df = group[group["label"] == "Normal"]
        if not baseline_df.empty:
            baseline_vals = baseline_df[feat_cols].median()
        else:
            # Fallback: 5% klatek o najniższej asymetrii
            baseline_vals = group.nsmallest(max(1, int(len(group) * 0.05)), "eye_aperture_asymmetry")[
                feat_cols].median()

        group[feat_cols] = group[feat_cols].values - baseline_vals.values
        normalized.append(group)
    return pd.concat(normalized, ignore_index=True)


def run_validation():
    print(f"--- WALIDACJA MODELU v1.4 ---")

    # 1. Ładowanie modelu i danych
    model_data = joblib.load(MODEL_PATH)
    ensemble = model_data["models"]
    m_normal = model_data["mult_normal"]
    m_strong = model_data["mult_strong"]

    df = pd.read_csv(VALIDATION_DATA_PATH)
    df["video_id"] = df["filename"].apply(lambda x: str(x).split("/")[0])

    # 2. Przetwarzanie (identyczne jak w training_v1_4.py)
    df = engineer_features_v1_4(df)
    df = add_temporal_features_validation(df)
    df_norm = apply_validation_normalization(df)

    # Pobranie cech (upewnij się, że kolejność kolumn zgadza się z modelem!)
    # W praktyce najlepiej zapisać listę kolumn w .pkl podczas treningu
    X_val = df_norm[ALL_FEATURE_COLS].values.astype(np.float32)
    y_true = df_norm["label"].values

    # 3. Soft Voting Ensemble [cite: 47]
    all_probs = []
    for name, model in ensemble:
        all_probs.append(model.predict_proba(X_val))

    # Uśrednianie prawdopodobieństw
    avg_probs = np.mean(all_probs, axis=0)

    # 4. Zastosowanie dynamicznych progów [cite: 52]
    p = avg_probs.copy()
    p[:, LABEL_ORDER.index("Normal")] *= m_normal
    p[:, LABEL_ORDER.index("StrongPalsy")] *= m_strong
    y_pred = [LABEL_ORDER[np.argmax(row)] for row in p]

    # 5. Raport końcowy
    print("\n[WYNIKI WALIDACJI]")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Macro: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_true, y_pred, target_names=LABEL_ORDER))


if __name__ == "__main__":
    run_validation()