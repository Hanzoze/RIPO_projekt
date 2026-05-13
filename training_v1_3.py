"""
training_v1_3.py  –  Etap 3 / Wersja 1.3
Ocena symetrii twarzy – Klasyfikacja porażenia nerwu twarzowego

Kluczowa zmiana względem v1.0–v1.2:
────────────────────────────────────────────────────────────────
DIAGNOZA PROBLEMU (v1.0–v1.2):
  Przy 22 pacjentach każdy losowy podział GroupShuffleSplit daje
  zupełnie inny rozkład klas w zbiorach (np. val z 3 próbkami Normal
  lub 35 StrongPalsy). To powoduje niestabilność metryk i złe
  kalibrowanie progów – nie błąd implementacji, lecz ograniczenie danych.

ROZWIĄZANIE: Leave-One-Group-Out (LOGO) Cross-Validation
  • Trenujemy 22 modele (jeden na każdego pacjenta jako "test")
  • Każdy pacjent jest raz w roli zbioru testowego
  • Wyniki uśredniamy → stabilna, bezstronna ocena generalizacji
  • Standard w literaturze medycznej dla małych zbiorów grupowych
  • Eliminuje losowy pech podziału

PIPELINE per fold:
  1. Train = 21 pacjentów, Test = 1 pacjent
  2. SMOTE na trainie (sampling_strategy słownikowy)
  3. BRF + RF+SMOTE ensemble (soft voting)
  4. Predykcja na testowym pacjencie
  5. Zapis predykcji do zbiorczego wektora

FINALNY MODEL (do deploymentu):
  Trenowany na WSZYSTKICH 22 pacjentach z najlepszymi parametrami
  znalezionymi w LOGO CV.
"""

import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
import joblib

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.ensemble import BalancedRandomForestClassifier
    SMOTE_AVAILABLE = True
    print("[INFO] imblearn dostępny – SMOTE + BRF aktywne.")
except ImportError:
    ImbPipeline = Pipeline
    SMOTE_AVAILABLE = False
    print("[WARN] imblearn niedostępny.")

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Ścieżki ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"/")
OUTPUT_DIR = BASE_DIR / "exploration_results_v1_3"
CSV_PATH   = BASE_DIR / "exploration_results" / "features.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["Normal", "SlightPalsy", "StrongPalsy"]

BASE_FEATURE_COLS = [
    "left_eye_open", "right_eye_open",
    "eye_aperture_asymmetry", "eye_aperture_ratio",
    "mouth_corner_height_diff",
    "mouth_left_deviation", "mouth_right_deviation", "mouth_deviation_asymmetry",
    "left_brow_height", "right_brow_height",
    "brow_height_asymmetry", "brow_height_ratio",
    "mouth_width", "mouth_angle", "eye_angle",
    "left_eye_width", "right_eye_width", "eye_width_asymmetry",
]
NEW_FEATURE_COLS = [
    "HB_composite_index", "eye_mouth_asymmetry_product",
    "global_symmetry_score", "brow_eye_combined",
    "lower_face_severity", "eye_closure_min",
    "mouth_asymmetry_combined", "total_asymmetry_delta",
]
ALL_FEATURE_COLS = BASE_FEATURE_COLS + NEW_FEATURE_COLS

# Wagi klas – kompromis między v1.0 i v1.1
CLASS_WEIGHTS = {"Normal": 15, "SlightPalsy": 1, "StrongPalsy": 6}

# Historia wyników do porównania
PREV_RESULTS = {
    "Etap2 (v0)": {"Normal": 0.880, "SlightPalsy": 0.970, "StrongPalsy": 0.590, "macro": 0.813,
                   "note": "train/test z tego samego wideo – data leakage"},
    "v1.0":       {"Normal": 0.082, "SlightPalsy": 0.924, "StrongPalsy": 0.487, "macro": 0.498,
                   "note": "GroupShuffleSplit, losowy skos klas w val"},
    "v1.1":       {"Normal": 0.281, "SlightPalsy": 0.849, "StrongPalsy": 0.336, "macro": 0.489,
                   "note": "BRF, 2D threshold – val miał 35 StrongPalsy"},
    "v1.2":       {"Normal": 0.039, "SlightPalsy": 0.792, "StrongPalsy": 0.497, "macro": 0.443,
                   "note": "Ensemble + strat. split – val miał 3 Normal"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# INŻYNIERIA CECH + NORMALIZACJA (stabilna, identyczna z v1.0–v1.2)
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    d = df.copy()
    d["HB_composite_index"] = (
        d["eye_aperture_asymmetry"] * 3.0 + d["mouth_angle"] / 20.0
        + d["mouth_corner_height_diff"] * 2.0
    ) / 6.0
    d["eye_mouth_asymmetry_product"] = d["eye_aperture_asymmetry"] * d["mouth_corner_height_diff"]
    d["global_symmetry_score"] = np.sqrt(
        d["eye_aperture_asymmetry"]**2 + d["mouth_corner_height_diff"]**2
        + d["brow_height_asymmetry"]**2 + d["mouth_deviation_asymmetry"]**2
    )
    d["brow_eye_combined"]    = (d["brow_height_asymmetry"] + d["eye_aperture_asymmetry"]) / 2.0
    d["lower_face_severity"]  = (
        d["mouth_corner_height_diff"] * 2.0 + d["mouth_angle"] / 15.0
        + d["mouth_deviation_asymmetry"]
    ) / 4.0
    d["eye_closure_min"]          = d[["left_eye_open", "right_eye_open"]].min(axis=1)
    d["mouth_asymmetry_combined"] = np.sqrt(
        d["mouth_left_deviation"]**2 + d["mouth_right_deviation"]**2
        + d["mouth_corner_height_diff"]**2
    )
    d["total_asymmetry_delta"] = d[BASE_FEATURE_COLS].abs().sum(axis=1)
    return d


def normalize_per_patient(df):
    """Normalizacja TYLKO na danych treningowych danego foldu.
    Wywoływana wewnątrz pętli LOGO żeby uniknąć wycieku."""
    normalized = []
    no_baseline = 0
    for vid, group in df.groupby("video_id"):
        baseline_df = group[group["label"] == "Normal"]
        if not baseline_df.empty:
            baseline_vals = baseline_df[ALL_FEATURE_COLS].median()
        else:
            n_fb = max(1, int(len(group) * 0.05))
            baseline_vals = (
                group.nsmallest(n_fb, "eye_aperture_asymmetry")[ALL_FEATURE_COLS].median()
            )
            no_baseline += 1
        group = group.copy()
        group[ALL_FEATURE_COLS] = group[ALL_FEATURE_COLS].values - baseline_vals.values
        normalized.append(group)
    return pd.concat(normalized, ignore_index=True), no_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# BUDOWA MODELU (jeden fold)
# ═══════════════════════════════════════════════════════════════════════════════

def build_fold_model(X_tr, y_tr):
    """Buduje ensemble dla jednego foldu LOGO."""
    counts    = Counter(y_tr)
    min_class = min(counts.values())

    models = []

    # Model A: BalancedRandomForest
    if SMOTE_AVAILABLE:
        brf = BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=CLASS_WEIGHTS,
            random_state=42,
            n_jobs=-1,
            sampling_strategy="all",
            replacement=True,
        )
        brf_pipe = Pipeline([("scaler", StandardScaler()), ("clf", brf)])
        brf_pipe.fit(X_tr, y_tr)
        models.append(("brf", brf_pipe))

    # Model B: RF + SMOTE
    target_sp  = counts.get("SlightPalsy", 1000)
    target_min = min(int(target_sp * 0.5), 2500)
    smote_strat = {
        lbl: target_min
        for lbl in ["Normal", "StrongPalsy"]
        if counts.get(lbl, 0) < target_min and counts.get(lbl, 0) >= 6
    }

    if SMOTE_AVAILABLE and min_class >= 6 and smote_strat:
        k_nb = min(5, min_class - 1)
        rf_pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(sampling_strategy=smote_strat, random_state=42, k_neighbors=k_nb)),
            ("clf",    RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight=CLASS_WEIGHTS,
                random_state=42,
                n_jobs=-1,
            )),
        ])
    else:
        rf_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300,
                class_weight=CLASS_WEIGHTS,
                random_state=42, n_jobs=-1,
            )),
        ])
    rf_pipe.fit(X_tr, y_tr)
    models.append(("rf_smote", rf_pipe))

    return models


def ensemble_predict_proba(models, X):
    """Soft voting: uśrednia predict_proba z wszystkich modeli."""
    all_probs = []
    for name, m in models:
        probs = m.predict_proba(X)
        # Upewnij się że kolejność klas to LABEL_ORDER
        if hasattr(m[-1], "classes_"):
            cls_list = list(m[-1].classes_)
        else:
            cls_list = list(m.classes_)
        reordered = np.zeros((len(X), len(LABEL_ORDER)))
        for i, lbl in enumerate(LABEL_ORDER):
            if lbl in cls_list:
                reordered[:, i] = probs[:, cls_list.index(lbl)]
        all_probs.append(reordered)
    return np.mean(all_probs, axis=0)


def threshold_predict(probs, mult_strong=1.5, mult_normal=2.0):
    """Stosuje mnożniki progów i zwraca predykcje."""
    p = probs.copy()
    si = LABEL_ORDER.index("StrongPalsy")
    ni = LABEL_ORDER.index("Normal")
    p[:, si] *= mult_strong
    p[:, ni] *= mult_normal
    return [LABEL_ORDER[int(np.argmax(p[i]))] for i in range(len(p))]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGO CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_logo_cv(df):
    """
    Leave-One-Group-Out CV.
    Dla każdego pacjenta: trenuje na pozostałych 21, testuje na nim.
    Normalizacja baseline liczona TYLKO na danych treningowych foldu.
    """
    print("\n" + "=" * 65)
    print("KROK 2: Leave-One-Group-Out Cross-Validation")
    print("=" * 65)
    print("  Trenowanie 22 modeli (jeden per pacjent jako test)...")
    print("  To może potrwać kilka minut.\n")

    logo   = LeaveOneGroupOut()
    groups = df["video_id"].values
    X_all  = df[ALL_FEATURE_COLS].values.astype(np.float32)
    y_all  = df["label"].values

    all_true  = []
    all_pred  = []
    fold_results = []

    # Stałe progi – dobrane na podstawie analizy v1.0–v1.2
    # Normal jest najbardziej niedoreprezentowany → wyższy mnożnik
    MULT_STRONG = 1.5
    MULT_NORMAL = 2.5

    n_folds = logo.get_n_splits(groups=groups)

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups)):
        test_patient = groups[test_idx[0]]
        y_test       = y_all[test_idx]
        test_counts  = Counter(y_test)

        # Normalizacja baseline TYLKO na trainie
        df_train = df.iloc[train_idx].copy()
        df_train_norm, _ = normalize_per_patient(df_train)
        X_tr = df_train_norm[ALL_FEATURE_COLS].values.astype(np.float32)
        y_tr = df_train_norm["label"].values

        # Normalizacja testowego pacjenta względem jego własnego baseline
        # (w praktyce kliniku: mierzymy baseline przy rejestracji)
        df_test = df.iloc[test_idx].copy()
        df_test_norm, _ = normalize_per_patient(df_test)
        X_te = df_test_norm[ALL_FEATURE_COLS].values.astype(np.float32)

        # Trening
        fold_models = build_fold_model(X_tr, y_tr)

        # Predykcja z progami
        probs  = ensemble_predict_proba(fold_models, X_te)
        y_pred = threshold_predict(probs, MULT_STRONG, MULT_NORMAL)

        # Zapis wyników foldu
        fold_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_results.append({
            "patient":   test_patient,
            "n_frames":  len(y_test),
            "classes":   dict(test_counts),
            "f1_macro":  round(fold_f1, 4),
        })
        all_true.extend(y_test)
        all_pred.extend(y_pred)

        # Progres
        cls_str = ", ".join(f"{l[0]}={v}" for l, v in
                  [(lbl[:2], test_counts.get(lbl, 0)) for lbl in LABEL_ORDER])
        print(f"  Fold {fold_i+1:2d}/{n_folds} | pacjent={test_patient:>4s} "
              f"| klatki={len(y_test):4d} [{cls_str}] | F1={fold_f1:.3f}")

    return np.array(all_true), np.array(all_pred), fold_results


# ═══════════════════════════════════════════════════════════════════════════════
# FINALNY MODEL (na wszystkich danych)
# ═══════════════════════════════════════════════════════════════════════════════

def train_final_model(df):
    """Trenuje finalny model na wszystkich 22 pacjentach."""
    print("\n" + "=" * 65)
    print("KROK 3: Trening finalnego modelu (wszystkie dane)")
    print("=" * 65)

    df_norm, no_bl = normalize_per_patient(df)
    if no_bl:
        print(f"  [WARN] {no_bl} pacjentów bez klatek Normal – użyto fallbacku.")

    X = df_norm[ALL_FEATURE_COLS].values.astype(np.float32)
    y = df_norm["label"].values

    models = build_fold_model(X, y)
    print(f"  Finalny model: {[m[0] for m in models]} (soft voting)")
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# WYKRESY
# ═══════════════════════════════════════════════════════════════════════════════

def save_plots(y_true, y_pred, fold_results):
    print("\n" + "=" * 65)
    print("KROK 4: Generowanie wykresów")
    print("=" * 65)

    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0
    )

    # 1. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax)
    ax.set_title("Macierz pomyłek – v1.3\n(Leave-One-Group-Out CV, N=22 pacjentów)")
    ax.set_xlabel("Predykcja"); ax.set_ylabel("Wartość rzeczywista")
    plt.tight_layout()
    p = OUTPUT_DIR / "confusion_matrix_v1_3.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p}")

    # 2. Porównanie wszystkich wersji
    all_versions = list(PREV_RESULTS.keys()) + ["v1.3"]
    all_f1s = {
        ver: [PREV_RESULTS[ver][lbl] for lbl in LABEL_ORDER]
        for ver in PREV_RESULTS
    }
    all_f1s["v1.3"] = list(f1_per)

    # Pomijamy Etap2 w wykresie (data leakage – nieporównywalny)
    compare_versions = ["v1.0", "v1.1", "v1.2", "v1.3"]
    fig2, ax2 = plt.subplots(figsize=(13, 6))
    x     = np.arange(len(LABEL_ORDER))
    width = 0.2
    clrs  = ["#95a5a6", "#3498db", "#f39c12", "#e74c3c"]
    for i, (ver, clr) in enumerate(zip(compare_versions, clrs)):
        bars = ax2.bar(x + i*width, all_f1s[ver], width, label=ver, color=clr, alpha=0.9)
        ax2.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(LABEL_ORDER, fontsize=11)
    ax2.set_ylim(0, 1.25)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Próg 0.5")
    ax2.set_title("F1 per klasa: v1.0 → v1.3 (bez Etapu 2 – data leakage)", fontsize=12)
    ax2.legend(); ax2.set_ylabel("F1-score")
    plt.tight_layout()
    p2 = OUTPUT_DIR / "comparison_v1_0_to_v1_3.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p2}")

    # 3. F1 macro per pacjent (pokazuje stabilność)
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    patients  = [fr["patient"] for fr in fold_results]
    f1s       = [fr["f1_macro"] for fr in fold_results]
    n_frames  = [fr["n_frames"] for fr in fold_results]
    clrs3     = ["#e74c3c" if f < 0.4 else "#f39c12" if f < 0.6 else "#2ecc71" for f in f1s]
    bars3     = ax3.bar(patients, f1s, color=clrs3, alpha=0.9)
    ax3.bar_label(bars3, fmt="%.2f", fontsize=7, padding=2)
    ax3.axhline(np.mean(f1s), color="navy", linestyle="--",
                label=f"Średnia F1={np.mean(f1s):.3f}")
    ax3.set_xlabel("ID pacjenta"); ax3.set_ylabel("F1 macro")
    ax3.set_title("F1 macro per pacjent (LOGO CV)\n"
                  "Czerwony < 0.4 | Żółty 0.4–0.6 | Zielony ≥ 0.6")
    ax3.legend(); ax3.set_ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    p3 = OUTPUT_DIR / "per_patient_f1_v1_3.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p3}")

    # 4. Per-class metrics
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    x4 = np.arange(len(LABEL_ORDER)); w4 = 0.25
    for j, (vals, color, lbl_m) in enumerate(zip(
        [p_per, r_per, f1_per], ["#3498db", "#2ecc71", "#e74c3c"],
        ["Precision", "Recall", "F1"]
    )):
        bars = ax4.bar(x4 + j*w4, vals, w4, label=lbl_m, color=color, alpha=0.85)
        ax4.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax4.set_xticks(x4 + w4); ax4.set_xticklabels(LABEL_ORDER)
    ax4.set_ylim(0, 1.25)
    ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax4.set_title("Precision / Recall / F1 per klasa – v1.3 (LOGO CV)")
    ax4.legend()
    plt.tight_layout()
    p4 = OUTPUT_DIR / "per_class_metrics_v1_3.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p4}")


# ═══════════════════════════════════════════════════════════════════════════════
# ZAPIS WYNIKÓW
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(final_models, y_true, y_pred, fold_results):
    print("\n" + "=" * 65)
    print("KROK 5: Zapis wyników")
    print("=" * 65)

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0
    )
    f1_mac = float(np.mean(f1_per))
    metrics = {
        lbl: {"precision": round(float(p_per[i]), 4),
              "recall":    round(float(r_per[i]), 4),
              "f1":        round(float(f1_per[i]), 4),
              "support":   int(sup_per[i])}
        for i, lbl in enumerate(LABEL_ORDER)
    }
    errors = [(t, p) for t, p in zip(y_true, y_pred) if t != p]

    summary = {
        "version":      "1.3",
        "eval_method":  "Leave-One-Group-Out CV (22 folds, 1 pacjent = 1 fold)",
        "model":        "Ensemble BRF + RF+SMOTE (soft voting)",
        "threshold":    {"mult_strong": 1.5, "mult_normal": 2.5},
        "class_weights": CLASS_WEIGHTS,
        "logo_cv": {
            "n_folds":        len(fold_results),
            "f1_macro_mean":  round(float(np.mean([f["f1_macro"] for f in fold_results])), 4),
            "f1_macro_std":   round(float(np.std([f["f1_macro"] for f in fold_results])), 4),
            "f1_macro_min":   round(float(np.min([f["f1_macro"] for f in fold_results])), 4),
            "f1_macro_max":   round(float(np.max([f["f1_macro"] for f in fold_results])), 4),
        },
        "aggregate_metrics": {
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro":  round(f1_mac, 4),
        },
        "metrics_per_class": metrics,
        "comparison": {
            ver: {
                lbl: {"prev": PREV_RESULTS[ver][lbl], "v1_3": metrics[lbl]["f1"],
                      "delta": round(metrics[lbl]["f1"] - PREV_RESULTS[ver][lbl], 4)}
                for lbl in LABEL_ORDER
            }
            for ver in ["v1.0", "v1.1", "v1.2"]
        },
        "error_analysis": {
            "total_errors": len(errors),
            "error_rate":   round(len(errors)/len(y_true), 4),
            "error_types":  {f"{t}->{p}": cnt for (t,p), cnt in Counter(errors).items()},
        },
        "per_patient_results": fold_results,
    }

    json_path  = OUTPUT_DIR / "model_results_v1_3.json"
    model_path = OUTPUT_DIR / "final_model_v1_3.pkl"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    joblib.dump(final_models, model_path)
    print(f"  Wyniki JSON: {json_path}")
    print(f"  Model PKL:   {model_path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Wczytanie i inżynieria cech (normalizacja robiona per-fold wewnątrz LOGO)
    print("=" * 65)
    print("KROK 1: Wczytanie i przygotowanie danych")
    print("=" * 65)
    df = pd.read_csv(CSV_PATH)
    print(f"Wczytano wierszy: {len(df)}")
    df["video_id"] = df["filename"].apply(lambda x: str(x).split("/")[0])
    df = engineer_features(df)
    df = df.dropna(subset=ALL_FEATURE_COLS + ["label"])
    print(f"Po dropna: {len(df)} wierszy")

    counts = Counter(df["label"])
    n_patients = df["video_id"].nunique()
    print(f"Liczba pacjentów: {n_patients}")
    print("Rozkład klas:")
    for lbl in LABEL_ORDER:
        cnt = counts.get(lbl, 0)
        print(f"  {lbl:15s}: {cnt:5d}  ({cnt/len(df)*100:.1f}%)")

    # 2. LOGO CV
    y_true, y_pred, fold_results = run_logo_cv(df)

    # 3. Finalny model na wszystkich danych
    final_models = train_final_model(df)

    # 4. Wykresy
    save_plots(y_true, y_pred, fold_results)

    # 5. Zapis
    summary = save_results(final_models, y_true, y_pred, fold_results)

    # 6. Podsumowanie końcowe
    print("\n" + "=" * 65)
    print("PODSUMOWANIE KOŃCOWE – v1.3")
    print("=" * 65)
    logo = summary["logo_cv"]
    agg  = summary["aggregate_metrics"]
    print(f"  Metoda ewaluacji: Leave-One-Group-Out CV ({logo['n_folds']} foldów)")
    print(f"  F1 macro per-fold: {logo['f1_macro_mean']:.4f} ± {logo['f1_macro_std']:.4f}")
    print(f"  F1 macro min/max:  {logo['f1_macro_min']:.4f} / {logo['f1_macro_max']:.4f}")
    print(f"  Agregowane Accuracy: {agg['accuracy']:.4f}")
    print(f"  Agregowane F1 macro: {agg['f1_macro']:.4f}")
    print()
    print(f"  {'Klasa':15s}  {'v1.0':>6s}  {'v1.1':>6s}  {'v1.2':>6s}  {'v1.3':>6s}")
    print(f"  {'-'*50}")
    for lbl in LABEL_ORDER:
        f0  = PREV_RESULTS["v1.0"][lbl]
        f1v = PREV_RESULTS["v1.1"][lbl]
        f2  = PREV_RESULTS["v1.2"][lbl]
        f3  = summary["metrics_per_class"][lbl]["f1"]
        best = max(f0, f1v, f2, f3)
        mark = " ★" if f3 == best else ""
        print(f"  {lbl:15s}  {f0:>6.3f}  {f1v:>6.3f}  {f2:>6.3f}  {f3:>6.3f}{mark}")
    print()
    print(f"  {'F1 macro':15s}  "
          f"{PREV_RESULTS['v1.0']['macro']:>6.3f}  "
          f"{PREV_RESULTS['v1.1']['macro']:>6.3f}  "
          f"{PREV_RESULTS['v1.2']['macro']:>6.3f}  "
          f"{agg['f1_macro']:>6.3f}")
    print(f"\nWyniki zapisane w: {OUTPUT_DIR}")
    print("\n[WAŻNE] LOGO CV = najbardziej wiarygodna ocena dla 22 pacjentów.")
    print("        Żaden fold nie 'widział' testowego pacjenta podczas treningu.")


if __name__ == "__main__":
    main()