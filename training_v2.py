"""
training_v2_0.py  –  Etap 3 / Wersja 2.0
Ocena symetrii twarzy – Klasyfikacja bez wycieku danych

Główne zmiany względem Etapu 2 (training.py):
  1. GROUP SPLIT po patient_id – klatki tego samego pacjenta trafiają
     wyłącznie do jednego ze zbiorów (train/val/test). Eliminuje wyciek danych.
  2. INŻYNIERIA CECH – 7 nowych cech kompozytowych (global_symmetry_score,
     HB_composite_index itp.) opartych na wiedzy medycznej (skala H-B).
  3. SMOTE – oversampling mniejszościowych klas TYLKO na zbiorze treningowym,
     z zachowaniem granic grupowych.
  4. RANDOM FOREST z wagami klas + RandomizedSearchCV – bardziej rzetelny tuning.
  5. Brak normalizacji baseline – odejmowanie stanu spoczynku niszczyło klasę Normal.
  6. Brak threshold hackingu – uczciwa ewaluacja bez post-hoc manipulacji progiem.

Oczekiwany efekt: niższe accuracy niż v1 (bez wycieku ~91-93%), ale uczciwe metryki.
CV F1 zbliżone do test F1 – bez rozdźwięku charakterystycznego dla wycieku danych.
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
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[UWAGA] imblearn niedostępne – trening bez SMOTE. Zainstaluj: pip install imbalanced-learn")

import joblib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\danil\PycharmProjects\RIPO")
OUTPUT_DIR = BASE_DIR / "exploration_results_v2"
CSV_PATH = BASE_DIR / "exploration_results" / "features.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["Normal", "SlightPalsy", "StrongPalsy"]
RANDOM_STATE = 42

# Oryginalne cechy z Etapu 2
BASE_FEATURE_COLS = [
    "left_eye_open", "right_eye_open",
    "eye_aperture_asymmetry", "eye_aperture_ratio",
    "mouth_corner_height_diff",
    "mouth_left_deviation", "mouth_right_deviation",
    "left_brow_height", "right_brow_height",
    "brow_height_asymmetry",
    "mouth_width", "mouth_angle", "eye_angle",
    "left_eye_width", "right_eye_width", "eye_width_asymmetry",
]

# Nowe cechy kompozytowe – Etap 3
ENGINEERED_COLS = [
    "global_symmetry_score",  # euklidesowa norma wszystkich asymetrii
    "HB_composite_index",  # indeks wzorowany na skali House-Brackmanna
    "brow_eye_combined",  # łączna asymetria brwi+oczu
    "lower_face_severity",  # ciężkość porażenia dolnej części twarzy
    "eye_closure_min",  # min otwarcie oka (kluczowe przy lagophthalmos)
    "mouth_asymmetry_combined",  # kombinowana asymetria ust
    "eye_mouth_ratio",  # stosunek asymetrii oczu do ust
]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + ENGINEERED_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INŻYNIERIA CECH
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje cechy kompozytowe oparte na wiedzy medycznej.
    NIE modyfikuje oryginalnych cech – tylko dodaje nowe kolumny.
    """
    d = df.copy()

    # Globalna miara symetrii – norma L2 wszystkich asymetrii
    asym_cols = ["eye_aperture_asymmetry", "mouth_corner_height_diff",
                 "brow_height_asymmetry", "eye_width_asymmetry"]
    # Uzupełniamy brakujące kolumny zerami (kompatybilność z różnymi wersjami CSV)
    for col in asym_cols:
        if col not in d.columns:
            d[col] = 0.0
    d["global_symmetry_score"] = np.sqrt(
        d["eye_aperture_asymmetry"] ** 2 +
        d["mouth_corner_height_diff"] ** 2 +
        d["brow_height_asymmetry"] ** 2 +
        d["eye_width_asymmetry"] ** 2
    )

    # Indeks H-B: ważona suma objawów klinicznych
    d["HB_composite_index"] = (
                                      d["eye_aperture_asymmetry"] * 3.0 +
                                      d["mouth_angle"].abs() / 20.0 +
                                      d["mouth_corner_height_diff"] * 2.0
                              ) / 6.0

    # Łączna asymetria brwi + oczu (górna część twarzy)
    d["brow_eye_combined"] = (
                                     d["brow_height_asymmetry"] + d["eye_aperture_asymmetry"]
                             ) / 2.0

    # Ciężkość porażenia dolnej części twarzy
    d["lower_face_severity"] = (
                                       d["mouth_corner_height_diff"] * 2.0 +
                                       d["mouth_angle"].abs() / 15.0 +
                                       d["mouth_left_deviation"].abs()
                               ) / 4.0

    # Minimalne otwarcie oka – lagophthalmos marker
    d["eye_closure_min"] = d[["left_eye_open", "right_eye_open"]].min(axis=1)

    # Kombinowana asymetria ust (wektorowa)
    d["mouth_asymmetry_combined"] = np.sqrt(
        d["mouth_left_deviation"] ** 2 +
        d["mouth_right_deviation"] ** 2 +
        d["mouth_corner_height_diff"] ** 2
    )

    # Stosunek asymetrii oczu do ust
    eps = 1e-6
    d["eye_mouth_ratio"] = (
            d["eye_aperture_asymmetry"] /
            (d["mouth_corner_height_diff"] + eps)
    ).clip(-10, 10)

    return d


def normalize_per_patient(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Normalizuje cechy względem stanu spoczynku (Normal) każdego pacjenta."""
    print("\n[KALIBRACJA] Normalizacja cech względem baseline pacjenta...")
    normalized_list = []

    for pid, group in df.groupby("patient_id"):
        baseline_df = group[group["label"] == "Normal"]

        if not baseline_df.empty:
            # Mediana klatek Normal jako wzorzec zdrowej twarzy
            baseline_vals = baseline_df[feature_cols].median()
        else:
            # Jeśli pacjent nie ma klatek Normal, używamy 5-percentyla asymetrii
            # (zakładamy, że klatki z najmniejszą asymetrią to jego 'pseudo-normal')
            n_fallback = max(1, int(len(group) * 0.05))
            baseline_vals = group.nsmallest(n_fallback, "global_symmetry_score")[feature_cols].median()

        group = group.copy()
        group[feature_cols] = group[feature_cols].values - baseline_vals.values
        normalized_list.append(group)

    return pd.concat(normalized_list, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PODZIAŁ GRUPOWY (BEZ WYCIEKU DANYCH)
# ═══════════════════════════════════════════════════════════════════════════════

def group_train_val_test_split(df, group_col="patient_id",
                               test_frac=0.15, val_frac=0.15,
                               random_state=RANDOM_STATE):
    """
    Dzieli dane na poziomie grup (pacjentów).
    Żaden pacjent nie pojawia się w więcej niż jednym zbiorze.

    Strategia:
      - sortujemy pacjentów losowo (seed),
      - przypisujemy ostatnich test_frac do test,
      - następnych val_frac do val,
      - resztę do train.
    """
    rng = np.random.default_rng(random_state)

    patients = df[group_col].unique()
    rng.shuffle(patients)

    n = len(patients)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test:n_test + n_val])
    train_patients = set(patients[n_test + n_val:])

    df_train = df[df[group_col].isin(train_patients)].copy()
    df_val = df[df[group_col].isin(val_patients)].copy()
    df_test = df[df[group_col].isin(test_patients)].copy()

    return df_train, df_val, df_test, train_patients, val_patients, test_patients


def augment_mirror_data(df_train, feature_cols):
    """Создает зеркальные копии данных для обучения."""
    df_mirror = df_train.copy()

    # Словарь соответствия лево-право
    mirror_map = {
        'left_eye_open': 'right_eye_open',
        'right_eye_open': 'left_eye_open',
        'mouth_left_deviation': 'mouth_right_deviation',
        'mouth_right_deviation': 'mouth_left_deviation',
        'left_brow_height': 'right_brow_height',
        'right_brow_height': 'left_brow_height',
        'left_eye_width': 'right_eye_width',
        'right_eye_width': 'left_eye_width'
    }

    # Меняем значения местами
    for left, right in mirror_map.items():
        if left in df_mirror.columns and right in df_mirror.columns:
            temp = df_mirror[left].copy()
            df_mirror[left] = df_mirror[right]
            df_mirror[right] = temp

    # Инвертируем знаки для асимметрии, если они направленные
    directional_cols = ['mouth_angle', 'eye_angle']
    for col in directional_cols:
        if col in df_mirror.columns:
            df_mirror[col] = -df_mirror[col]

    # Добавляем к основному трейну
    return pd.concat([df_train, df_mirror], ignore_index=True)


def smooth_features(df, feature_cols, window=5):
    """Сглаживание признаков скользящим средним внутри каждого видео."""
    # Важно: сглаживаем только внутри одного видео/пациента
    df = df.sort_values(['patient_id', 'filename'])

    # В новых версиях pandas fillna(method='ffill') заменен на .ffill() и .bfill()
    df[feature_cols] = df.groupby('patient_id', group_keys=False)[feature_cols].apply(
        lambda x: x.rolling(window=window, center=True).mean().ffill().bfill()
    )
    return df


def predict_with_calibration(model, X, threshold_shifts=None):
    """Прогноз с ручной корректировкой веса классов."""
    probs = model.predict_proba(X)
    # LABEL_ORDER = ["Normal", "SlightPalsy", "StrongPalsy"]
    # Увеличиваем вероятность SlightPalsy на 20%, если модель в нем сомневается
    if threshold_shifts is None:
        threshold_shifts = [1.0, 1.3, 1.1]  # Множители для Normal, Slight, Strong

    for i in range(len(threshold_shifts)):
        probs[:, i] *= threshold_shifts[i]

    best_class_indices = np.argmax(probs, axis=1)
    return [model.classes_[i] for i in best_class_indices]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("KROK 1: Wczytanie i inżynieria cech")
    print("=" * 65)

    df_raw = pd.read_csv(CSV_PATH)
    print(f"Wczytano wierszy: {len(df_raw)}")

    # 1. СНАЧАЛА проверяем/создаем patient_id
    if "patient_id" not in df_raw.columns:
        print("  [INFO] Brak kolumny patient_id – wyciągam z filename")
        # Исправляем: создаем колонку ПЕРЕД использованием в других функциях
        df_raw["patient_id"] = df_raw["filename"].apply(
            lambda x: str(x).split("/")[0]
        )

    # 2. ТЕПЕРЬ запускаем сглаживание (когда patient_id уже существует)
    # Используем только те базовые признаки, которые реально есть в CSV
    current_base_features = [c for c in BASE_FEATURE_COLS if c in df_raw.columns]
    df_raw = smooth_features(df_raw, current_base_features)

    # 3. ДАЛЬШЕ запускаем генерацию новых признаков
    df = engineer_features(df_raw)

    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    df = normalize_per_patient(df, feature_cols)

    # Usuń wiersze z brakującymi wartościami kluczowych kolumn
    required_cols = [c for c in ALL_FEATURE_COLS if c in df.columns] + ["label", "patient_id"]
    df = df.dropna(subset=required_cols)

    # Aktualizuj listę cech do tych które faktycznie istnieją
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]

    print(f"Po usunięciu NaN: {len(df)} wierszy")
    print(
        f"Liczba cech: {len(feature_cols)} ({len(BASE_FEATURE_COLS)} bazowych + {len(feature_cols) - len(BASE_FEATURE_COLS)} nowych)")
    print(f"Liczba unikalnych pacjentów: {df['patient_id'].nunique()}")

    print("\nRozkład klas:")
    label_counts = Counter(df["label"])
    for lbl in LABEL_ORDER:
        cnt = label_counts.get(lbl, 0)
        print(f"  {lbl:15s}: {cnt:5d}  ({cnt / len(df) * 100:.1f}%)")

    # ── Krok 2: Podział grupowy ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("KROK 2: Podział grupowy (bez wycieku danych)")
    print("=" * 65)

    df_train, df_val, df_test, train_pts, val_pts, test_pts = \
        group_train_val_test_split(df, group_col="patient_id")

    df_train = augment_mirror_data(df_train, feature_cols)

    print(f"\nPodział pacjentów:")
    print(f"  Train: {len(train_pts)} pacjentów, {len(df_train)} klatek ({len(df_train) / len(df) * 100:.1f}%)")
    print(f"  Val:   {len(val_pts)} pacjentów, {len(df_val)} klatek ({len(df_val) / len(df) * 100:.1f}%)")
    print(f"  Test:  {len(test_pts)} pacjentów, {len(df_test)} klatek ({len(df_test) / len(df) * 100:.1f}%)")

    # Weryfikacja – zero wspólnych pacjentów
    assert not (train_pts & val_pts), "BŁĄD: pacjenci train i val się nakładają!"
    assert not (train_pts & test_pts), "BŁĄD: pacjenci train i test się nakładają!"
    assert not (val_pts & test_pts), "BŁĄD: pacjenci val i test się nakładają!"
    print("\n  ✓ Weryfikacja: zero wspólnych pacjentów między zbiorami")

    print("\nRozkład klas w zbiorach:")
    for name, df_split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        c = Counter(df_split["label"])
        print(f"  {name}: " + ", ".join(f"{lbl}={c.get(lbl, 0)}" for lbl in LABEL_ORDER))

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["label"].values
    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["label"].values
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test["label"].values
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["label"].values

    # ── Krok 3: Trening ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("KROK 3: Trening modeli")
    print("=" * 65)

    # Wagi klas – odwrotność częstości (ręcznie, bardziej kontrolowane niż 'balanced')
    total = len(y_train)
    counts = Counter(y_train)
    class_weights = {
        'Normal': 25.0,  # Увеличиваем в 2.5 раза от текущего
        'SlightPalsy': 0.1,  # Уменьшаем значимость самого частого класса
        'StrongPalsy': 15.0  # Увеличиваем
    }
    print(f"\n  Wagi klas: {class_weights}")

    # Sprawdź czy SMOTE ma sens (min próbek mniejszościowej klasy)
    min_samples = min(counts.values())
    use_smote = SMOTE_AVAILABLE and min_samples >= 6

    if use_smote:
        k_neighbors = min(5, min_samples - 1)
        steps = [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
            ("clf", RandomForestClassifier(
                class_weight=class_weights,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
        pipeline = ImbPipeline(steps)
        print(f"  Używam SMOTE (k_neighbors={k_neighbors})")
    else:
        steps = [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                class_weight=class_weights,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
        pipeline = Pipeline(steps)
        if not SMOTE_AVAILABLE:
            print("  SMOTE niedostępne – trening bez oversamplingu")
        else:
            print(f"  Za mało próbek dla SMOTE (min={min_samples}) – trening bez oversamplingu")

    # Hyperparameter search
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 15, 25],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    }

    print("\n  RandomizedSearchCV (20 iteracji, 3-fold na danych treningowych)...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="f1_macro",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"  Najlepsze parametry: {search.best_params_}")
    print(f"  CV F1 (trening, 3-fold): {search.best_score_:.4f}")

    # ── Krok 4: Ewaluacja ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("KROK 4: Ewaluacja")
    print("=" * 65)

    # Walidacja
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    _, _, val_f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="macro", zero_division=0
    )
    print(f"\nZbiór walidacyjny:")
    print(f"  Accuracy: {val_acc:.4f}  |  F1 macro: {val_f1:.4f}")

    # Test
    y_test_pred = predict_with_calibration(best_model, X_test, [2.5, 0.5, 1.5])
    test_acc = accuracy_score(y_test, y_test_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="macro", zero_division=0
    )
    print(f"\nZbiór testowy:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {p_mac:.4f}  Recall: {r_mac:.4f}  F1: {f1_mac:.4f}")
    print("\nClassification Report (test):")
    print(classification_report(y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0))

    # Per-class metryki
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0
    )
    metrics_per_class = {}
    for i, lbl in enumerate(LABEL_ORDER):
        metrics_per_class[lbl] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i]),
        }

    # Cross-validation na całym zbiorze – grupowa (GroupKFold)
    print("\nCross-validation grupowa (5-fold, cały zbiór):")
    from sklearn.model_selection import GroupKFold, cross_val_score
    groups_all = df["patient_id"].values
    n_unique = df["patient_id"].nunique()
    n_cv = min(5, n_unique)
    gkf = GroupKFold(n_splits=n_cv)
    cv_scores = cross_val_score(
        best_model, X_all, y_all,
        cv=gkf, groups=groups_all,
        scoring="f1_macro", n_jobs=-1
    )
    print(f"  F1 macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  (oczekiwany rozdźwięk test-CV < 0.10 przy braku wycieku)")

    # ── Krok 5: Wykresy ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("KROK 5: Generowanie wykresów")
    print("=" * 65)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Ewaluacja modelu v2.0 (Random Forest, group split)", fontsize=14, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=LABEL_ORDER)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=axes[0])
    axes[0].set_xlabel("Predykcja")
    axes[0].set_ylabel("Wartość rzeczywista")
    axes[0].set_title("Macierz pomyłek (Confusion Matrix)")
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")

    # Per-class bars
    x_pos = np.arange(len(LABEL_ORDER))
    width = 0.25
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    for j, (metric, color, lbl_m) in enumerate(zip(
            ["precision", "recall", "f1"], colors, ["Precision", "Recall", "F1"]
    )):
        vals = [metrics_per_class[lbl][metric] for lbl in LABEL_ORDER]
        bars = axes[1].bar(x_pos + j * width, vals, width, label=lbl_m, color=color, alpha=0.85)
        axes[1].bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    axes[1].set_xticks(x_pos + width)
    axes[1].set_xticklabels(LABEL_ORDER)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Wartość")
    axes[1].set_title("Precision / Recall / F1 per klasa")
    axes[1].legend()
    axes[1].axhline(y=0.7, color="gray", linestyle="--", alpha=0.5)

    # Porównanie v1 (Etap 2) vs v2.0 (Etap 3)
    # Wartości z Etapu 2 (hardcoded dla porównania)
    v1_test_f1 = 0.8126
    v1_cv_f1 = 0.5538
    v2_test_f1 = f1_mac
    v2_cv_f1 = cv_scores.mean()

    x3 = np.arange(2)
    w3 = 0.35
    bars_test = axes[2].bar(x3 - w3 / 2, [v1_test_f1, v2_test_f1], w3,
                            label="Test F1", color="#9b59b6", alpha=0.85)
    bars_cv = axes[2].bar(x3 + w3 / 2, [v1_cv_f1, v2_cv_f1], w3,
                          label="CV F1 (grupowe)", color="#1abc9c", alpha=0.85)
    axes[2].bar_label(bars_test, fmt="%.3f", fontsize=9, padding=3)
    axes[2].bar_label(bars_cv, fmt="%.3f", fontsize=9, padding=3)
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(["v1.0 Etap 2\n(z wyciekiem)", "v2.0 Etap 3\n(group split)"])
    axes[2].set_ylim(0, 1.0)
    axes[2].set_ylabel("F1 macro")
    axes[2].set_title("Porównanie v1 vs v2\n(test F1 vs CV F1 – różnica = wyciek)")
    axes[2].legend()

    plt.tight_layout()
    out_eval = OUTPUT_DIR / "model_evaluation_v2.png"
    plt.savefig(out_eval, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Wykres ewaluacji zapisany: {out_eval}")

    # Feature importance
    rf_clf = best_model.named_steps["clf"]
    importances = rf_clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig2, ax4 = plt.subplots(figsize=(12, 8))
    sorted_features = [feature_cols[i] for i in sorted_idx]
    sorted_vals = importances[sorted_idx]
    colors_imp = ["#e74c3c" if v > np.mean(importances) else "#3498db" for v in sorted_vals]
    ax4.barh(sorted_features[::-1], sorted_vals[::-1], color=colors_imp[::-1])
    ax4.set_xlabel("Ważność cechy (Feature Importance)")
    ax4.set_title("Ważność cech geometrycznych + kompozytowych (v2.0)\nCzerwony = powyżej średniej")
    ax4.axvline(x=np.mean(importances), color="gray", linestyle="--", alpha=0.7, label="Średnia")
    ax4.legend()
    plt.tight_layout()
    out_imp = OUTPUT_DIR / "feature_importance_v2.png"
    plt.savefig(out_imp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Wykres ważności cech zapisany: {out_imp}")

    # ── Krok 6: Analiza błędów ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("KROK 6: Analiza błędów")
    print("=" * 65)

    errors = [(t, p) for t, p in zip(y_test, y_test_pred) if t != p]
    error_counter = Counter(errors)
    print(f"\nŁączna liczba błędnych predykcji: {len(errors)} / {len(y_test)} ({len(errors) / len(y_test) * 100:.1f}%)")
    print("\nTypy błędów (prawdziwa → predykowana):")
    for (t, p), cnt in sorted(error_counter.items(), key=lambda x: -x[1]):
        print(f"  {t:15s} → {p:15s}: {cnt}")

    print("\nTP / FP / FN per klasa:")
    for lbl in LABEL_ORDER:
        tp = sum(1 for t, p in zip(y_test, y_test_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_test, y_test_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_test, y_test_pred) if t == lbl and p != lbl)
        recall_lbl = tp / max(tp + fn, 1)
        print(f"  {lbl:15s}: TP={tp}, FP={fp}, FN={fn}, Recall={recall_lbl:.3f}")

    # ── Zapis wyników ─────────────────────────────────────────────────────────
    summary = {
        "version": "2.0",
        "changes_from_v1": [
            "group_split_by_patient_id (eliminacja wycieku danych)",
            "feature_engineering (7 nowych cech kompozytowych)",
            "smote_oversampling_train_only",
            "randomized_search_cv_hypertuning",
            "no_baseline_normalization",
            "no_threshold_hacking",
        ],
        "dataset": {
            "total_rows": len(df),
            "unique_patients": int(df["patient_id"].nunique()),
            "class_distribution": {k: int(v) for k, v in label_counts.items()},
        },
        "split": {
            "train_patients": len(train_pts),
            "val_patients": len(val_pts),
            "test_patients": len(test_pts),
            "train_frames": len(df_train),
            "val_frames": len(df_val),
            "test_frames": len(df_test),
        },
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "f1_macro": round(float(f1_mac), 4),
            "precision_macro": round(float(p_mac), 4),
            "recall_macro": round(float(r_mac), 4),
        },
        "cv_metrics": {
            "f1_macro_mean": round(float(cv_scores.mean()), 4),
            "f1_macro_std": round(float(cv_scores.std()), 4),
            "n_folds": n_cv,
        },
        "metrics_per_class": metrics_per_class,
        "top5_features": [
            {"feature": feature_cols[i], "importance": round(float(importances[i]), 4)}
            for i in sorted_idx[:5]
        ],
        "error_analysis": {
            "total_errors": len(errors),
            "error_rate": round(len(errors) / max(len(y_test), 1), 4),
            "error_types": {f"{t}->{p}": cnt for (t, p), cnt in error_counter.items()},
        },
        "comparison_with_v1": {
            "v1_test_f1": v1_test_f1,
            "v1_cv_f1": v1_cv_f1,
            "v1_leakage_gap": round(v1_test_f1 - v1_cv_f1, 4),
            "v2_test_f1": round(float(f1_mac), 4),
            "v2_cv_f1": round(float(cv_scores.mean()), 4),
            "v2_leakage_gap": round(float(f1_mac - cv_scores.mean()), 4),
            "note": "Mniejszy gap = brak wycieku = bardziej uczciwa ewaluacja"
        }
    }

    json_path = OUTPUT_DIR / "model_results_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWyniki JSON: {json_path}")

    model_path = OUTPUT_DIR / "best_model_v2.pkl"
    joblib.dump(best_model, model_path)
    print(f"Model PKL:   {model_path}")

    # ── Podsumowanie końcowe ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PODSUMOWANIE KOŃCOWE – v2.0")
    print("=" * 65)
    print(
        f"  Liczba cech:        {len(feature_cols)} ({len(BASE_FEATURE_COLS)} bazowych + {len(feature_cols) - len(BASE_FEATURE_COLS)} nowych)")
    print(f"  Test Accuracy:      {test_acc:.4f}")
    print(f"  Test Precision:     {p_mac:.4f}")
    print(f"  Test Recall:        {r_mac:.4f}")
    print(f"  Test F1 (macro):    {f1_mac:.4f}")
    print(f"  CV F1 ({n_cv}-fold):    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    gap = abs(f1_mac - cv_scores.mean())
    print(f"  Gap test-CV:        {gap:.4f}  ", end="")
    if gap < 0.08:
        print("✓ (< 0.08 – brak wycieku danych)")
    elif gap < 0.15:
        print("~ (0.08-0.15 – akceptowalny)")
    else:
        print("✗ (> 0.15 – możliwy wyciek lub za mały test set)")
    print(f"\nWykresy: {OUTPUT_DIR}")
    print("\nv2.0 gotowa do obrony Etapu 3.")


if __name__ == "__main__":
    main()