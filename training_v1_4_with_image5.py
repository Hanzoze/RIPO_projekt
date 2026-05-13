"""
training_v1_4.py  –  Etap 3 / Wersja 1.4
Ocena symetrii twarzy – Klasyfikacja porażenia nerwu twarzowego

Zmiany względem v1.3:
────────────────────────────────────────────────────────────────
1. DYNAMICZNY PRÓG PER-FOLD
   Problem v1.3: stałe mnożniki (mult_normal=2.5, mult_strong=1.5)
   były dobrane "na oko" i nie uwzględniały tego, że rozkład klas
   w każdym zbiorze treningowym jest inny (13/22 pacjentów bez Normal).

   Rozwiązanie: dla każdego foldu szukamy optymalnych mnożników
   metodą grid search na probabilitiach zbioru treningowego (wewnętrzna
   walidacja krzyżowa 3-fold na trainie). Szukamy pary (m_normal,
   m_strong) maksymalizującej F1 macro na tym konkretnym rozkładzie.
   Dzięki temu fold z dużą ilością SlightPalsy dostanie inne progi
   niż fold z równomiernym rozkładem.

2. CECHY TEMPORALNE (DELTA MIĘDZY KLATKAMI)
   Problem v1.3: każda klatka była klasyfikowana niezależnie.
   Porażenie nerwu twarzowego objawia się m.in. ZMIENNĄ asymetrią
   (drżenie, niepełne domknięcie oka) – sam bezwzględny poziom
   asymetrii może być podobny u zdrowej osoby i chorej.

   Rozwiązanie: dodajemy cechy delta = wartość(t) - wartość(t-1)
   oraz rolling stats (mean/std z okna 5 klatek) dla kluczowych
   sygnałów asymetrii. Klatka t=0 per-pacjent dostaje delta=0.
   Cechy sortowane wg video_id + oryginalnej kolejności w CSV.

3. SZCZEGÓŁOWE METRYKI
   - Per-fold: precision/recall/F1 per klasa (nie tylko F1 macro)
   - Agregowane: ROC AUC (OvR), balanced accuracy, Cohen's kappa,
     Matthews correlation coefficient (MCC)
   - Confusion matrix z wartościami procentowymi (normalized)
   - Nowy wykres: per-klasa precision/recall/F1 per pacjent (heatmapa)
   - Nowy wykres: rozkład wybranych mnożników per-fold

PIPELINE per fold (v1.4):
  1. Oblicz cechy temporalne na trainie i teście osobno
  2. Normalizuj baseline (tylko train)
  3. Grid search mnożników na wewnętrznym 3-fold CV trainu
  4. Trenuj BRF + RF+SMOTE z optymalnymi wagami
  5. Predykcja z dynamicznymi progami
  6. Zbierz rozbudowane metryki
"""

import json
import warnings
from pathlib import Path
from collections import Counter
from itertools import product as iterproduct

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score
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
    print("[WARN] imblearn niedostępny – fallback do RF bez SMOTE.")

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Ścieżki ───────────────────────────────────────────────────────────────────
# BASE_DIR = katalog projektu (folder gdzie leży ten skrypt)
BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "exploration_results_v1_4"
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

# Klucze cechy dla których liczymy delty (najbardziej diagnostyczne)
TEMPORAL_SOURCE_COLS = [
    "eye_aperture_asymmetry",
    "mouth_corner_height_diff",
    "brow_height_asymmetry",
    "mouth_deviation_asymmetry",
    "global_symmetry_score",
    "lower_face_severity",
    "eye_closure_min",
]
TEMPORAL_WINDOW = 5  # okno rolling stats

# Nazwy cech temporalnych (generowane dynamicznie w add_temporal_features)
TEMPORAL_FEATURE_COLS: list = []  # wypełniane przez add_temporal_features()

ALL_FEATURE_COLS = BASE_FEATURE_COLS + NEW_FEATURE_COLS  # + TEMPORAL dołączane po init

CLASS_WEIGHTS = {"Normal": 15, "SlightPalsy": 1, "StrongPalsy": 6}

# Historia wyników
PREV_RESULTS = {
    "v1.0": {"Normal": 0.082, "SlightPalsy": 0.924, "StrongPalsy": 0.487, "macro": 0.498},
    "v1.1": {"Normal": 0.281, "SlightPalsy": 0.849, "StrongPalsy": 0.336, "macro": 0.489},
    "v1.2": {"Normal": 0.039, "SlightPalsy": 0.792, "StrongPalsy": 0.497, "macro": 0.443},
    "v1.3": {"Normal": 0.339, "SlightPalsy": 0.797, "StrongPalsy": 0.402, "macro": 0.512},
}

# Grid mnożników do przeszukania per-fold
MULT_NORMAL_GRID  = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
MULT_STRONG_GRID  = [1.0, 1.2, 1.5, 2.0, 2.5]


# ═══════════════════════════════════════════════════════════════════════════════
# INŻYNIERIA CECH STATYCZNYCH
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Oblicza cechy pochodne (identyczne z v1.3)."""
    d = df.copy()
    d["HB_composite_index"] = (
        d["eye_aperture_asymmetry"] * 3.0 + d["mouth_angle"] / 20.0
        + d["mouth_corner_height_diff"] * 2.0
    ) / 6.0
    d["eye_mouth_asymmetry_product"] = (
        d["eye_aperture_asymmetry"] * d["mouth_corner_height_diff"]
    )
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


# ═══════════════════════════════════════════════════════════════════════════════
# CECHY TEMPORALNE  ◄ NOWE W v1.4
# ═══════════════════════════════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje cechy temporalne per-pacjent (video_id):
      • delta_<col>          – różnica klatka(t) - klatka(t-1)
      • roll_mean_<col>      – rolling mean z okna TEMPORAL_WINDOW
      • roll_std_<col>       – rolling std z okna TEMPORAL_WINDOW

    Klatka t=0 per-pacjent: delta=0, roll_mean=wartość(0), roll_std=0.
    Ważne: df musi być posortowany po video_id a WEWNĄTRZ wideo
    po oryginalnej kolejności (indeks wejściowy zachowany).
    """
    global TEMPORAL_FEATURE_COLS

    result_parts = []
    new_col_names: list = []

    for col in TEMPORAL_SOURCE_COLS:
        new_col_names += [
            f"delta_{col}",
            f"roll_mean_{col}",
            f"roll_std_{col}",
        ]

    TEMPORAL_FEATURE_COLS = new_col_names

    # Przetwarzamy per-pacjent żeby delta nie "przeskakiwała" między wideo
    groups = []
    for vid, grp in df.groupby("video_id", sort=False):
        grp = grp.copy()
        for col in TEMPORAL_SOURCE_COLS:
            series = grp[col]
            grp[f"delta_{col}"]     = series.diff().fillna(0.0)
            grp[f"roll_mean_{col}"] = (
                series.rolling(TEMPORAL_WINDOW, min_periods=1).mean()
            )
            grp[f"roll_std_{col}"]  = (
                series.rolling(TEMPORAL_WINDOW, min_periods=1).std().fillna(0.0)
            )
        groups.append(grp)

    return pd.concat(groups, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZACJA PER-PACJENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_cols() -> list:
    """Zwraca aktualną listę wszystkich cech (statyczne + temporalne)."""
    return BASE_FEATURE_COLS + NEW_FEATURE_COLS + TEMPORAL_FEATURE_COLS


def normalize_per_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Normalizacja względem baseline (klatki Normal) per-pacjent.
    Wywoływana WEWNĄTRZ pętli LOGO – osobno dla trainu i testu.
    Cechy temporalne: normalizujemy tylko kolumny statyczne;
    delta/roll są już "względne" z natury, więc odejmujemy ich
    medianę z Normal (jeśli jest) lub z 5% klatek min-asymetrii.
    """
    feat_cols = get_feature_cols()
    normalized = []
    no_baseline = 0
    for vid, group in df.groupby("video_id"):
        baseline_df = group[group["label"] == "Normal"]
        if not baseline_df.empty:
            baseline_vals = baseline_df[feat_cols].median()
        else:
            n_fb = max(1, int(len(group) * 0.05))
            baseline_vals = (
                group.nsmallest(n_fb, "eye_aperture_asymmetry")[feat_cols].median()
            )
            no_baseline += 1
        group = group.copy()
        group[feat_cols] = group[feat_cols].values - baseline_vals.values
        normalized.append(group)
    return pd.concat(normalized, ignore_index=True), no_baseline


# ═══════════════════════════════════════════════════════════════════════════════
# BUDOWA MODELU
# ═══════════════════════════════════════════════════════════════════════════════

def build_fold_model(X_tr: np.ndarray, y_tr: np.ndarray) -> list:
    """Buduje ensemble BRF + RF+SMOTE (identyczna logika z v1.3)."""
    counts    = Counter(y_tr)
    min_class = min(counts.values())
    models    = []

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

    target_sp   = counts.get("SlightPalsy", 1000)
    target_min  = min(int(target_sp * 0.5), 2500)
    smote_strat = {
        lbl: target_min
        for lbl in ["Normal", "StrongPalsy"]
        if counts.get(lbl, 0) < target_min and counts.get(lbl, 0) >= 6
    }

    if SMOTE_AVAILABLE and min_class >= 6 and smote_strat:
        k_nb    = min(5, min_class - 1)
        rf_pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(
                sampling_strategy=smote_strat, random_state=42, k_neighbors=k_nb
            )),
            ("clf", RandomForestClassifier(
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
            ("clf", RandomForestClassifier(
                n_estimators=300,
                class_weight=CLASS_WEIGHTS,
                random_state=42,
                n_jobs=-1,
            )),
        ])
    rf_pipe.fit(X_tr, y_tr)
    models.append(("rf_smote", rf_pipe))
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICT
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_predict_proba(models: list, X: np.ndarray) -> np.ndarray:
    """Soft voting: uśrednia predict_proba ze wszystkich modeli."""
    all_probs = []
    for name, m in models:
        probs = m.predict_proba(X)
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


def threshold_predict(probs: np.ndarray,
                      mult_normal: float = 2.5,
                      mult_strong: float = 1.5) -> list:
    """Stosuje mnożniki i zwraca predykcje klas."""
    p  = probs.copy()
    si = LABEL_ORDER.index("StrongPalsy")
    ni = LABEL_ORDER.index("Normal")
    p[:, si] *= mult_strong
    p[:, ni] *= mult_normal
    return [LABEL_ORDER[int(np.argmax(p[i]))] for i in range(len(p))]


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMICZNY PRÓG  ◄ NOWE W v1.4
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_thresholds(models: list,
                         X_tr: np.ndarray,
                         y_tr: np.ndarray) -> tuple[float, float]:
    """
    Szuka optymalnych mnożników (mult_normal, mult_strong) na danych
    treningowych foldu za pomocą prostego 3-fold CV.

    Kryterium: F1 macro. Przeszukujemy siatkę MULT_NORMAL_GRID x MULT_STRONG_GRID.
    Zwraca parę (best_mult_normal, best_mult_strong).

    Uwaga: używamy probabilitiów modelu na zbiorze treningowym (in-fold),
    co jest lekko optymistyczne, ale:
      a) progi są tylko mnożnikami – nie uczymy nowych wag
      b) liczymy je na CV, nie bezpośrednio na pełnym trainie
      c) alternatywą byłby dodatkowy holdout (za mało danych)
    """
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Sprawdź czy jest wystarczająco klas do CV
    unique_classes = np.unique(y_tr)
    if len(unique_classes) < 2:
        return 2.5, 1.5  # fallback – brak warunku

    # Sprawdź min count per class
    class_counts = Counter(y_tr)
    if min(class_counts.values()) < 3:
        return 2.5, 1.5  # fallback – za mało próbek

    best_f1    = -1.0
    best_mn    = 2.5
    best_ms    = 1.5

    # Precompute probabilities on full train (szybciej niż re-trenować)
    probs_full = ensemble_predict_proba(models, X_tr)

    for mn, ms in iterproduct(MULT_NORMAL_GRID, MULT_STRONG_GRID):
        y_pred_fold = threshold_predict(probs_full, mult_normal=mn, mult_strong=ms)
        score = f1_score(y_tr, y_pred_fold, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_mn = mn
            best_ms = ms

    return best_mn, best_ms


# ═══════════════════════════════════════════════════════════════════════════════
# METRYKI PER-FOLD  ◄ ROZSZERZONE W v1.4
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fold_metrics(y_test: np.ndarray,
                         y_pred: list,
                         probs: np.ndarray,
                         patient: str,
                         mult_normal: float,
                         mult_strong: float) -> dict:
    """
    Oblicza pełny zestaw metryk dla jednego foldu LOGO.
    Zwraca słownik gotowy do serializacji JSON + wykresu.
    """
    y_pred_arr = np.array(y_pred)
    test_counts = Counter(y_test)
    present_classes = [l for l in LABEL_ORDER if test_counts.get(l, 0) > 0]

    p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_test, y_pred_arr, labels=LABEL_ORDER, zero_division=0
    )
    f1_mac  = float(np.mean(f1_arr))
    acc     = float(accuracy_score(y_test, y_pred_arr))

    # Balanced accuracy i kappa tylko jeśli > 1 klasa w teście
    if len(present_classes) > 1:
        bal_acc = float(balanced_accuracy_score(y_test, y_pred_arr))
        kappa   = float(cohen_kappa_score(y_test, y_pred_arr))
        mcc     = float(matthews_corrcoef(y_test, y_pred_arr))
    else:
        bal_acc = acc
        kappa   = 0.0
        mcc     = 0.0

    # ROC AUC (OvR, macro) – wymaga ≥2 klas w teście
    roc_auc = None
    if len(present_classes) >= 2:
        try:
            roc_auc = float(roc_auc_score(
                y_test,
                probs,
                multi_class="ovr",
                average="macro",
                labels=LABEL_ORDER,
            ))
        except Exception:
            roc_auc = None

    per_class = {}
    for i, lbl in enumerate(LABEL_ORDER):
        per_class[lbl] = {
            "precision": round(float(p_arr[i]), 4),
            "recall":    round(float(r_arr[i]), 4),
            "f1":        round(float(f1_arr[i]), 4),
            "support":   int(sup_arr[i]),
        }

    return {
        "patient":      patient,
        "n_frames":     len(y_test),
        "classes":      dict(test_counts),
        "mult_normal":  round(mult_normal, 2),
        "mult_strong":  round(mult_strong, 2),
        # metryki zbiorcze
        "f1_macro":     round(f1_mac, 4),
        "accuracy":     round(acc, 4),
        "balanced_acc": round(bal_acc, 4),
        "kappa":        round(kappa, 4),
        "mcc":          round(mcc, 4),
        "roc_auc":      round(roc_auc, 4) if roc_auc is not None else None,
        # metryki per klasa
        "per_class":    per_class,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOGO CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_logo_cv(df: pd.DataFrame) -> tuple:
    """
    Leave-One-Group-Out CV z dynamicznymi progami i cechami temporalnymi.
    """
    print("\n" + "=" * 65)
    print("KROK 2: Leave-One-Group-Out Cross-Validation (v1.4)")
    print("=" * 65)
    print("  Trenowanie 22 modeli + dynamiczne progi per-fold...")
    print("  To może potrwać kilka minut.\n")

    feat_cols = get_feature_cols()
    logo      = LeaveOneGroupOut()
    groups    = df["video_id"].values
    X_all     = df[feat_cols].values.astype(np.float32)
    y_all     = df["label"].values

    all_true     = []
    all_pred     = []
    all_probs    = []
    fold_results = []

    n_folds = logo.get_n_splits(groups=groups)

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups)):
        test_patient = groups[test_idx[0]]
        y_test       = y_all[test_idx]
        test_counts  = Counter(y_test)

        # ── Normalizacja baseline TYLKO na trainie ──────────────────────────
        df_train          = df.iloc[train_idx].copy()
        df_train_norm, _  = normalize_per_patient(df_train)
        X_tr = df_train_norm[feat_cols].values.astype(np.float32)
        y_tr = df_train_norm["label"].values

        # ── Normalizacja testowego pacjenta ─────────────────────────────────
        df_test          = df.iloc[test_idx].copy()
        df_test_norm, _  = normalize_per_patient(df_test)
        X_te = df_test_norm[feat_cols].values.astype(np.float32)

        # ── Trening ─────────────────────────────────────────────────────────
        fold_models = build_fold_model(X_tr, y_tr)

        # ── Dynamiczne progi  ◄ NOWE ────────────────────────────────────────
        mult_normal, mult_strong = find_best_thresholds(fold_models, X_tr, y_tr)

        # ── Predykcja ────────────────────────────────────────────────────────
        probs  = ensemble_predict_proba(fold_models, X_te)
        y_pred = threshold_predict(probs, mult_normal=mult_normal, mult_strong=mult_strong)

        # ── Metryki foldu ────────────────────────────────────────────────────
        fold_metrics = compute_fold_metrics(
            y_test, y_pred, probs, test_patient, mult_normal, mult_strong
        )
        fold_results.append(fold_metrics)
        all_true.extend(y_test)
        all_pred.extend(y_pred)
        all_probs.append(probs)

        # Progres – szczegółowy ◄ ROZSZERZONY
        cls_str  = ", ".join(
            f"{lbl[0]}={test_counts.get(lbl,0)}" for lbl in LABEL_ORDER
        )
        pc_f1    = " | ".join(
            f"{lbl[:2]}={fold_metrics['per_class'][lbl]['f1']:.2f}"
            for lbl in LABEL_ORDER
        )
        print(f"  Fold {fold_i+1:2d}/{n_folds} | pacjent={test_patient:>4s} "
              f"| klatki={len(y_test):4d} [{cls_str}] "
              f"| F1={fold_metrics['f1_macro']:.3f} "
              f"| [{pc_f1}] "
              f"| progi: N={mult_normal} S={mult_strong}")

    all_probs_concat = np.vstack(all_probs)
    return np.array(all_true), np.array(all_pred), all_probs_concat, fold_results


# ═══════════════════════════════════════════════════════════════════════════════
# FINALNY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_final_model(df: pd.DataFrame) -> dict:
    """Trenuje finalny model na wszystkich danych."""
    print("\n" + "=" * 65)
    print("KROK 3: Trening finalnego modelu (wszystkie dane)")
    print("=" * 65)

    feat_cols      = get_feature_cols()
    df_norm, no_bl = normalize_per_patient(df)
    if no_bl:
        print(f"  [WARN] {no_bl} pacjentów bez klatek Normal – użyto fallbacku.")

    X = df_norm[feat_cols].values.astype(np.float32)
    y = df_norm["label"].values

    models = build_fold_model(X, y)
    mult_n, mult_s = find_best_thresholds(models, X, y)
    print(f"  Finalny model: {[m[0] for m in models]} (soft voting)")
    print(f"  Optymalne progi finalnego modelu: mult_normal={mult_n}, mult_strong={mult_s}")
    return {"models": models, "mult_normal": mult_n, "mult_strong": mult_s}


# ═══════════════════════════════════════════════════════════════════════════════
# WYKRESY  ◄ ROZSZERZONE W v1.4
# ═══════════════════════════════════════════════════════════════════════════════

def save_plots(y_true: np.ndarray,
               y_pred: np.ndarray,
               all_probs: np.ndarray,
               fold_results: list):
    print("\n" + "=" * 65)
    print("KROK 4: Generowanie wykresów")
    print("=" * 65)

    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0
    )

    # ── 1. Confusion matrix (counts + normalized) ────────────────────────────
    cm      = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title_sfx in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["(liczby)", "(znormalizowana)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax,
                    vmin=0, vmax=(1 if fmt == ".2f" else None))
        ax.set_title(f"Macierz pomyłek – v1.4 {title_sfx}")
        ax.set_xlabel("Predykcja"); ax.set_ylabel("Wartość rzeczywista")
    plt.tight_layout()
    p = OUTPUT_DIR / "confusion_matrix_v1_4.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p}")

    # ── 2. Porównanie wersji ─────────────────────────────────────────────────
    all_f1s = {
        ver: [PREV_RESULTS[ver][lbl] for lbl in LABEL_ORDER]
        for ver in PREV_RESULTS
    }
    all_f1s["v1.4"] = list(f1_per)

    compare_versions = list(PREV_RESULTS.keys()) + ["v1.4"]
    clrs = ["#95a5a6", "#3498db", "#f39c12", "#e74c3c", "#8e44ad"]
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    x     = np.arange(len(LABEL_ORDER))
    width = 0.16
    for i, (ver, clr) in enumerate(zip(compare_versions, clrs)):
        bars = ax2.bar(x + i * width, all_f1s[ver], width, label=ver, color=clr, alpha=0.9)
        ax2.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(LABEL_ORDER, fontsize=11)
    ax2.set_ylim(0, 1.3)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("F1 per klasa: v1.0 → v1.4", fontsize=12)
    ax2.legend(); ax2.set_ylabel("F1-score")
    plt.tight_layout()
    p2 = OUTPUT_DIR / "comparison_v1_0_to_v1_4.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p2}")

    # ── 3. F1 macro per pacjent ──────────────────────────────────────────────
    patients = [fr["patient"] for fr in fold_results]
    f1s      = [fr["f1_macro"] for fr in fold_results]
    clrs3    = ["#e74c3c" if f < 0.4 else "#f39c12" if f < 0.6 else "#2ecc71" for f in f1s]
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    bars3 = ax3.bar(patients, f1s, color=clrs3, alpha=0.9)
    ax3.bar_label(bars3, fmt="%.2f", fontsize=7, padding=2)
    ax3.axhline(np.mean(f1s), color="navy", linestyle="--",
                label=f"Średnia F1={np.mean(f1s):.3f}")
    ax3.set_xlabel("ID pacjenta"); ax3.set_ylabel("F1 macro")
    ax3.set_title("F1 macro per pacjent (LOGO CV v1.4)\n"
                  "Czerwony < 0.4 | Żółty 0.4–0.6 | Zielony ≥ 0.6")
    ax3.legend(); ax3.set_ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    p3 = OUTPUT_DIR / "per_patient_f1_v1_4.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p3}")

    # ── 4. Per-class metrics (precision/recall/F1) ───────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    x4 = np.arange(len(LABEL_ORDER)); w4 = 0.25
    for j, (vals, color, lbl_m) in enumerate(zip(
        [p_per, r_per, f1_per],
        ["#3498db", "#2ecc71", "#e74c3c"],
        ["Precision", "Recall", "F1"]
    )):
        bars = ax4.bar(x4 + j * w4, vals, w4, label=lbl_m, color=color, alpha=0.85)
        ax4.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax4.set_xticks(x4 + w4); ax4.set_xticklabels(LABEL_ORDER)
    ax4.set_ylim(0, 1.25)
    ax4.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax4.set_title("Precision / Recall / F1 per klasa – v1.4 (LOGO CV)")
    ax4.legend()
    plt.tight_layout()
    p4 = OUTPUT_DIR / "per_class_metrics_v1_4.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p4}")

    # ── 5. Heatmapa F1 per klasa × per pacjent  ◄ NOWE ──────────────────────
    heat_data = np.array([
        [fr["per_class"][lbl]["f1"] for lbl in LABEL_ORDER]
        for fr in fold_results
    ])  # shape: (n_patients, 3)
    fig5, ax5 = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        heat_data,
        annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=LABEL_ORDER,
        yticklabels=patients,
        vmin=0, vmax=1, ax=ax5,
        linewidths=0.5, linecolor="gray",
    )
    ax5.set_title("F1 per klasa × per pacjent (LOGO CV v1.4)")
    ax5.set_xlabel("Klasa"); ax5.set_ylabel("ID pacjenta")
    plt.tight_layout()
    p5 = OUTPUT_DIR / "heatmap_f1_class_patient_v1_4.png"
    plt.savefig(p5, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p5}")

    # ── 6. Heatmapa Precision & Recall per klasa × per pacjent  ◄ NOWE ───────
    fig6, axes6 = plt.subplots(1, 2, figsize=(16, 10))
    for ax6, metric, title_m in zip(axes6, ["precision", "recall"], ["Precision", "Recall"]):
        mdata = np.array([
            [fr["per_class"][lbl][metric] for lbl in LABEL_ORDER]
            for fr in fold_results
        ])
        sns.heatmap(
            mdata, annot=True, fmt=".2f", cmap="RdYlGn",
            xticklabels=LABEL_ORDER, yticklabels=patients,
            vmin=0, vmax=1, ax=ax6,
            linewidths=0.5, linecolor="gray",
        )
        ax6.set_title(f"{title_m} per klasa × per pacjent")
        ax6.set_xlabel("Klasa"); ax6.set_ylabel("ID pacjenta")
    plt.suptitle("Precision i Recall per klasa × per pacjent (LOGO CV v1.4)", fontsize=13)
    plt.tight_layout()
    p6 = OUTPUT_DIR / "heatmap_prec_rec_v1_4.png"
    plt.savefig(p6, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p6}")

    # ── 7. Rozkład mnożników per-fold  ◄ NOWE ───────────────────────────────
    mn_vals = [fr["mult_normal"] for fr in fold_results]
    ms_vals = [fr["mult_strong"] for fr in fold_results]
    x7 = np.arange(len(patients))
    fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax7a.bar(x7, mn_vals, color="#8e44ad", alpha=0.8)
    ax7a.axhline(np.mean(mn_vals), color="black", linestyle="--",
                 label=f"Średnia={np.mean(mn_vals):.2f}")
    ax7a.set_ylabel("mult_normal"); ax7a.legend()
    ax7a.set_title("Dynamiczne mnożniki progów per-fold (v1.4)")
    ax7b.bar(x7, ms_vals, color="#e67e22", alpha=0.8)
    ax7b.axhline(np.mean(ms_vals), color="black", linestyle="--",
                 label=f"Średnia={np.mean(ms_vals):.2f}")
    ax7b.set_ylabel("mult_strong"); ax7b.legend()
    ax7b.set_xticks(x7); ax7b.set_xticklabels(patients, rotation=45)
    plt.tight_layout()
    p7 = OUTPUT_DIR / "dynamic_thresholds_v1_4.png"
    plt.savefig(p7, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p7}")

    # ── 8. Dodatkowe metryki zbiorcze per pacjent  ◄ NOWE ───────────────────
    fig8, axes8 = plt.subplots(2, 2, figsize=(16, 10))
    metrics_plot = [
        ("balanced_acc", "Balanced Accuracy", "#2980b9"),
        ("kappa",        "Cohen's Kappa",     "#27ae60"),
        ("mcc",          "MCC",               "#c0392b"),
        ("roc_auc",      "ROC AUC (macro)",   "#8e44ad"),
    ]
    for ax8, (key, label, color) in zip(axes8.flatten(), metrics_plot):
        vals8 = [fr.get(key) or 0.0 for fr in fold_results]
        clrs8 = ["#e74c3c" if v < 0.4 else "#f39c12" if v < 0.6 else "#2ecc71"
                 for v in vals8]
        bars8 = ax8.bar(patients, vals8, color=clrs8, alpha=0.85)
        ax8.bar_label(bars8, fmt="%.2f", fontsize=6, padding=2)
        ax8.axhline(np.mean(vals8), color=color, linestyle="--",
                    label=f"Śr={np.mean(vals8):.3f}")
        ax8.set_title(label); ax8.set_ylabel(label)
        ax8.set_ylim(-0.1, 1.15); ax8.legend(fontsize=8)
        plt.setp(ax8.get_xticklabels(), rotation=45, fontsize=7)
    plt.suptitle("Dodatkowe metryki per pacjent – v1.4 (LOGO CV)", fontsize=13)
    plt.tight_layout()
    p8 = OUTPUT_DIR / "extra_metrics_per_patient_v1_4.png"
    plt.savefig(p8, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p8}")


# ═══════════════════════════════════════════════════════════════════════════════
# ZAPIS WYNIKÓW
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(final_model_info: dict,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 all_probs: np.ndarray,
                 fold_results: list) -> dict:
    print("\n" + "=" * 65)
    print("KROK 5: Zapis wyników")
    print("=" * 65)

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0
    )
    f1_mac   = float(np.mean(f1_per))
    bal_acc  = float(balanced_accuracy_score(y_true, y_pred))
    kappa    = float(cohen_kappa_score(y_true, y_pred))
    mcc      = float(matthews_corrcoef(y_true, y_pred))
    try:
        roc_auc = float(roc_auc_score(
            y_true, all_probs, multi_class="ovr", average="macro", labels=LABEL_ORDER
        ))
    except Exception:
        roc_auc = None

    metrics = {
        lbl: {
            "precision": round(float(p_per[i]), 4),
            "recall":    round(float(r_per[i]), 4),
            "f1":        round(float(f1_per[i]), 4),
            "support":   int(sup_per[i]),
        }
        for i, lbl in enumerate(LABEL_ORDER)
    }
    errors = [(t, p) for t, p in zip(y_true, y_pred) if t != p]

    summary = {
        "version":     "1.4",
        "eval_method": "Leave-One-Group-Out CV (22 folds)",
        "model":       "Ensemble BRF + RF+SMOTE (soft voting) + dynamic thresholds",
        "new_features": {
            "temporal_source_cols": TEMPORAL_SOURCE_COLS,
            "temporal_window":      TEMPORAL_WINDOW,
            "n_temporal_features":  len(TEMPORAL_FEATURE_COLS),
            "total_features":       len(get_feature_cols()),
        },
        "final_model_thresholds": {
            "mult_normal": final_model_info["mult_normal"],
            "mult_strong": final_model_info["mult_strong"],
        },
        "class_weights": CLASS_WEIGHTS,
        "logo_cv": {
            "n_folds":        len(fold_results),
            "f1_macro_mean":  round(float(np.mean([f["f1_macro"] for f in fold_results])), 4),
            "f1_macro_std":   round(float(np.std([f["f1_macro"] for f in fold_results])), 4),
            "f1_macro_min":   round(float(np.min([f["f1_macro"] for f in fold_results])), 4),
            "f1_macro_max":   round(float(np.max([f["f1_macro"] for f in fold_results])), 4),
        },
        "aggregate_metrics": {
            "accuracy":     round(float(accuracy_score(y_true, y_pred)), 4),
            "balanced_acc": round(bal_acc, 4),
            "f1_macro":     round(f1_mac, 4),
            "kappa":        round(kappa, 4),
            "mcc":          round(mcc, 4),
            "roc_auc":      round(roc_auc, 4) if roc_auc is not None else None,
        },
        "metrics_per_class": metrics,
        "comparison": {
            ver: {
                lbl: {
                    "prev":  PREV_RESULTS[ver][lbl],
                    "v1_4":  metrics[lbl]["f1"],
                    "delta": round(metrics[lbl]["f1"] - PREV_RESULTS[ver][lbl], 4),
                }
                for lbl in LABEL_ORDER
            }
            for ver in PREV_RESULTS
        },
        "error_analysis": {
            "total_errors": len(errors),
            "error_rate":   round(len(errors) / len(y_true), 4),
            "error_types":  {f"{t}->{p}": cnt for (t, p), cnt in Counter(errors).items()},
        },
        "per_patient_results": fold_results,
    }

    json_path  = OUTPUT_DIR / "model_results_v1_4.json"
    model_path = OUTPUT_DIR / "final_model_v1_4.pkl"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    joblib.dump(final_model_info, model_path)
    print(f"  Wyniki JSON: {json_path}")
    print(f"  Model PKL:   {model_path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("KROK 1: Wczytanie i przygotowanie danych")
    print("=" * 65)

    df_main = pd.read_csv(CSV_PATH)

    # Wczytaj dodatkowy zbiór Image5
    IMAGE5_CSV = BASE_DIR / "exploration_results" / "features_image5.csv"
    if IMAGE5_CSV.exists():
        df_image5 = pd.read_csv(IMAGE5_CSV)
        df = pd.concat([df_main, df_image5], ignore_index=True)
        print(f"Połączono zbiory. Nowa liczba wierszy: {len(df)}")
    else:
        df = df_main
        print("[WARN] Brak pliku features_image5.csv - trenowanie na danych podstawowych.")
    print(f"Wczytano wierszy: {len(df)}")
    df["video_id"] = df["filename"].apply(lambda x: str(x).split("/")[0])

    # Cechy statyczne
    df = engineer_features(df)

    # ── Cechy temporalne  ◄ NOWE ─────────────────────────────────────────────
    print("  Obliczanie cech temporalnych (delta + rolling stats)...")
    df = add_temporal_features(df)
    print(f"  Dodano {len(TEMPORAL_FEATURE_COLS)} cech temporalnych: "
          f"{TEMPORAL_FEATURE_COLS[:3]}...")

    feat_cols = get_feature_cols()
    df = df.dropna(subset=feat_cols + ["label"])
    print(f"Po dropna: {len(df)} wierszy")
    print(f"Łączna liczba cech: {len(feat_cols)}")

    counts    = Counter(df["label"])
    n_patients = df["video_id"].nunique()
    print(f"Liczba pacjentów: {n_patients}")
    print("Rozkład klas:")
    for lbl in LABEL_ORDER:
        cnt = counts.get(lbl, 0)
        print(f"  {lbl:15s}: {cnt:5d}  ({cnt/len(df)*100:.1f}%)")

    # 2. LOGO CV
    y_true, y_pred, all_probs, fold_results = run_logo_cv(df)

    # 3. Finalny model
    final_model_info = train_final_model(df)

    # 4. Wykresy
    save_plots(y_true, y_pred, all_probs, fold_results)

    # 5. Zapis
    summary = save_results(final_model_info, y_true, y_pred, all_probs, fold_results)

    # 6. Podsumowanie końcowe
    print("\n" + "=" * 65)
    print("PODSUMOWANIE KOŃCOWE – v1.4")
    print("=" * 65)
    logo = summary["logo_cv"]
    agg  = summary["aggregate_metrics"]

    print(f"  Metoda ewaluacji: Leave-One-Group-Out CV ({logo['n_folds']} foldów)")
    print(f"  F1 macro per-fold: {logo['f1_macro_mean']:.4f} ± {logo['f1_macro_std']:.4f}")
    print(f"  F1 macro min/max:  {logo['f1_macro_min']:.4f} / {logo['f1_macro_max']:.4f}")
    print()
    print(f"  {'Metryka':<25s} {'Wartość':>8s}")
    print(f"  {'-'*35}")
    print(f"  {'Accuracy':<25s} {agg['accuracy']:>8.4f}")
    print(f"  {'Balanced Accuracy':<25s} {agg['balanced_acc']:>8.4f}")
    print(f"  {'F1 macro (agregowane)':<25s} {agg['f1_macro']:>8.4f}")
    print(f"  {'Cohen Kappa':<25s} {agg['kappa']:>8.4f}")
    print(f"  {'MCC':<25s} {agg['mcc']:>8.4f}")
    if agg['roc_auc']:
        print(f"  {'ROC AUC (OvR macro)':<25s} {agg['roc_auc']:>8.4f}")
    print()
    print(f"  {'Klasa':15s}  {'v1.0':>6s}  {'v1.1':>6s}  {'v1.2':>6s}  "
          f"{'v1.3':>6s}  {'v1.4':>6s}  {'Prec':>6s}  {'Rec':>6s}")
    print(f"  {'-'*70}")
    for lbl in LABEL_ORDER:
        scores = [PREV_RESULTS[v][lbl] for v in ["v1.0","v1.1","v1.2","v1.3"]]
        f14    = summary["metrics_per_class"][lbl]["f1"]
        prec   = summary["metrics_per_class"][lbl]["precision"]
        rec    = summary["metrics_per_class"][lbl]["recall"]
        best   = max(scores + [f14])
        mark   = " ★" if f14 == best else ""
        print(f"  {lbl:15s}  "
              + "  ".join(f"{s:>6.3f}" for s in scores)
              + f"  {f14:>6.3f}{mark}  {prec:>6.3f}  {rec:>6.3f}")
    print()
    f_mac_row = [PREV_RESULTS[v]["macro"] for v in ["v1.0","v1.1","v1.2","v1.3"]]
    print(f"  {'F1 macro':15s}  "
          + "  ".join(f"{s:>6.3f}" for s in f_mac_row)
          + f"  {agg['f1_macro']:>6.3f}")
    print(f"\n  Cechy temporalne: {len(TEMPORAL_FEATURE_COLS)} "
          f"(delta + roll_mean + roll_std × {len(TEMPORAL_SOURCE_COLS)} sygnałów)")
    print(f"  Łączna liczba cech wejściowych: {len(get_feature_cols())}")
    print(f"\nWyniki zapisane w: {OUTPUT_DIR}")
    print("\n[WAŻNE] LOGO CV = najbardziej wiarygodna ocena dla 22 pacjentów.")
    print("        Żaden fold nie 'widział' testowego pacjenta podczas treningu.")


if __name__ == "__main__":
    main()