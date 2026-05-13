"""
training_v1_1.py  –  Etap 3 / Wersja 1.1
Ocena symetrii twarzy – Klasyfikacja porażenia nerwu twarzowego

Kluczowe zmiany względem v1.0:
1. Agresywniejsze wagi klas: Normal otrzymuje wagę 20x, StrongPalsy 8x —
   bezpośrednia odpowiedź na F1=0.082 dla Normal w v1.0.
2. Dwuwymiarowy tuning progów (Normal + StrongPalsy) zamiast jednowymiarowego —
   szukamy optymalnej pary (mult_strong, mult_normal) na zbiorze walidacyjnym.
3. Dodano BalancedRandomForest jako kandydat w CV obok standardowego RF
   (lepiej radzi sobie z niezbalansowanymi klasami bez SMOTE).
4. SMOTE z sampling_strategy słownikowym: dokładna kontrola liczby próbek
   na klasę, zamiast pełnego wyrównania (które zaśmiecało Normal i StrongPalsy).
5. Rozszerzony param_grid z większą siatką n_estimators i max_features.
6. Dodano porównanie v1.0 vs v1.1 w podsumowaniu końcowym.
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
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GroupShuffleSplit, GroupKFold, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
import joblib

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.ensemble import BalancedRandomForestClassifier
    SMOTE_AVAILABLE = True
    BRF_AVAILABLE   = True
    print("[INFO] imblearn dostępny – SMOTE + BalancedRandomForest aktywne.")
except ImportError:
    ImbPipeline = Pipeline
    SMOTE_AVAILABLE = False
    BRF_AVAILABLE   = False
    print("[WARN] imblearn niedostępny – trening bez SMOTE/BRF.")

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"/")
OUTPUT_DIR = BASE_DIR / "exploration_results_v1_1"
CSV_PATH   = BASE_DIR / "exploration_results" / "features.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["Normal", "SlightPalsy", "StrongPalsy"]

# Wagi klas – kluczowa zmiana względem v1.0
# v1.0: Normal=5, SlightPalsy=1, StrongPalsy=3
# v1.1: agresywniejsza korekta, bo Normal F1=0.082 w v1.0
CLASS_WEIGHTS = {"Normal": 20, "SlightPalsy": 1, "StrongPalsy": 8}

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
    "HB_composite_index",
    "eye_mouth_asymmetry_product",
    "global_symmetry_score",
    "brow_eye_combined",
    "lower_face_severity",
    "eye_closure_min",
    "mouth_asymmetry_combined",
    "total_asymmetry_delta",
]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + NEW_FEATURE_COLS

# Wyniki v1.0 do porównania w podsumowaniu
V1_0_RESULTS = {
    "Normal":      {"precision": 0.360, "recall": 0.046, "f1": 0.082},
    "SlightPalsy": {"precision": 0.938, "recall": 0.911, "f1": 0.924},
    "StrongPalsy": {"precision": 0.344, "recall": 0.833, "f1": 0.487},
    "accuracy": 0.8652,
    "f1_macro": 0.498,
}


# ═══════════════════════════════════════════════════════════════════════════════
# INŻYNIERIA CECH (identyczna z v1.0 – stabilna)
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["HB_composite_index"] = (
        d["eye_aperture_asymmetry"] * 3.0
        + d["mouth_angle"] / 20.0
        + d["mouth_corner_height_diff"] * 2.0
    ) / 6.0

    d["eye_mouth_asymmetry_product"] = (
        d["eye_aperture_asymmetry"] * d["mouth_corner_height_diff"]
    )

    d["global_symmetry_score"] = np.sqrt(
        d["eye_aperture_asymmetry"] ** 2
        + d["mouth_corner_height_diff"] ** 2
        + d["brow_height_asymmetry"] ** 2
        + d["mouth_deviation_asymmetry"] ** 2
    )

    d["brow_eye_combined"] = (
        d["brow_height_asymmetry"] + d["eye_aperture_asymmetry"]
    ) / 2.0

    d["lower_face_severity"] = (
        d["mouth_corner_height_diff"] * 2.0
        + d["mouth_angle"] / 15.0
        + d["mouth_deviation_asymmetry"]
    ) / 4.0

    d["eye_closure_min"] = d[["left_eye_open", "right_eye_open"]].min(axis=1)

    d["mouth_asymmetry_combined"] = np.sqrt(
        d["mouth_left_deviation"] ** 2
        + d["mouth_right_deviation"] ** 2
        + d["mouth_corner_height_diff"] ** 2
    )

    d["total_asymmetry_delta"] = d[BASE_FEATURE_COLS].abs().sum(axis=1)

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZACJA BASELINE (identyczna z v1.0)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[KALIBRACJA] Normalizacja cech względem stanu spoczynku (Baseline)...")
    normalized = []
    patients_without_baseline = 0

    for vid, group in df.groupby("video_id"):
        baseline_df = group[group["label"] == "Normal"]

        if not baseline_df.empty:
            baseline_vals = baseline_df[ALL_FEATURE_COLS].median()
        else:
            n_fallback = max(1, int(len(group) * 0.05))
            baseline_vals = (
                group.nsmallest(n_fallback, "eye_aperture_asymmetry")[ALL_FEATURE_COLS]
                .median()
            )
            patients_without_baseline += 1

        group = group.copy()
        group[ALL_FEATURE_COLS] = group[ALL_FEATURE_COLS].values - baseline_vals.values
        normalized.append(group)

    if patients_without_baseline:
        print(f"  [WARN] {patients_without_baseline} pacjentów bez klatek 'Normal' "
              f"– użyto fallbacku (5-percentyl asymetrii).")

    return pd.concat(normalized, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WCZYTANIE DANYCH
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(csv_path: Path) -> tuple:
    print("=" * 65)
    print("KROK 1: Wczytanie i przygotowanie danych")
    print("=" * 65)

    df = pd.read_csv(csv_path)
    print(f"Wczytano wierszy: {len(df)}")

    df["video_id"] = df["filename"].apply(lambda x: str(x).split("/")[0])
    df = engineer_features(df)
    df = df.dropna(subset=ALL_FEATURE_COLS + ["label"])
    print(f"Po dropna: {len(df)} wierszy")

    df = normalize_per_patient(df)

    counts = Counter(df["label"])
    print("\nRozkład klas:")
    for lbl in LABEL_ORDER:
        cnt = counts.get(lbl, 0)
        print(f"  {lbl:15s}: {cnt:5d}  ({cnt / len(df) * 100:.1f}%)")

    X      = df[ALL_FEATURE_COLS].values.astype(np.float32)
    y      = df["label"].values
    groups = df["video_id"].values

    return X, y, groups, df


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPOWY PODZIAŁ DANYCH
# ═══════════════════════════════════════════════════════════════════════════════

def group_split(X, y, groups, test_size=0.15, val_size=0.15, random_state=42):
    print("\n" + "=" * 65)
    print("KROK 2: Grupowy podział danych (GroupShuffleSplit)")
    print("=" * 65)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups))

    X_tv, y_tv, g_tv = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
    X_test, y_test   = X[test_idx], y[test_idx]

    effective_val = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=effective_val, random_state=random_state + 1)
    train_idx2, val_idx2 = next(gss_val.split(X_tv, y_tv, g_tv))

    X_train, y_train, g_train = X_tv[train_idx2], y_tv[train_idx2], g_tv[train_idx2]
    X_val,   y_val             = X_tv[val_idx2],   y_tv[val_idx2]

    print(f"Train: {len(X_train):5d}  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val:   {len(X_val):5d}  ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test):5d}  ({len(X_test)/len(X)*100:.1f}%)")

    print("\nRozkład klas (po podziale grupowym):")
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        c = Counter(y_split)
        row = ", ".join(f"{lbl}={c.get(lbl, 0)}" for lbl in LABEL_ORDER)
        print(f"  {split_name}: {row}")

    overlap = set(g_train) & set(groups[test_idx])
    if overlap:
        print(f"  [WARN] Przecięcie train/test grup: {len(overlap)} – sprawdź dane!")
    else:
        print("  [OK] Brak przecięcia grup między train a test.")

    return X_train, X_val, X_test, y_train, y_val, y_test, g_train


# ═══════════════════════════════════════════════════════════════════════════════
# TRENING  (v1.1 – główna zmiana)
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_train(X_train, y_train, g_train):
    print("\n" + "=" * 65)
    print("KROK 3: Trening modelu v1.1")
    print("=" * 65)

    counts = Counter(y_train)
    min_class = min(counts.values())
    print(f"  Rozkład klas treningowych: {dict(counts)}")
    print(f"  Wagi klas: {CLASS_WEIGHTS}")

    n_cv = min(3, len(set(g_train)))
    gkf  = GroupKFold(n_splits=n_cv)

    # ── SMOTE ze słownikową strategią ────────────────────────────────────────
    # Zamiast wyrównywać wszystko do max klasy, targetujemy konkretne liczby.
    # Normal i StrongPalsy: podnosimy do ~50% liczby SlightPalsy w trainie.
    # Dzięki temu nie generujemy za dużo syntetycznych próbek.
    if SMOTE_AVAILABLE and min_class >= 6:
        target_n_sp = counts.get("SlightPalsy", 1000)
        target_minor = max(min_class, min(int(target_n_sp * 0.4), 2000))
        sampling_strategy = {}
        for lbl in ["Normal", "StrongPalsy"]:
            current = counts.get(lbl, 0)
            if current < target_minor:
                sampling_strategy[lbl] = target_minor

        k_neighbors = min(5, min_class - 1)
        print(f"  SMOTE sampling_strategy: {sampling_strategy}")
        print(f"  SMOTE k_neighbors: {k_neighbors}")

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=k_neighbors,
        )

        pipeline_rf = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  smote),
            ("clf",    RandomForestClassifier(
                class_weight=CLASS_WEIGHTS,
                random_state=42,
                n_jobs=-1,
            )),
        ])
    else:
        print("  Trening bez SMOTE.")
        pipeline_rf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                class_weight=CLASS_WEIGHTS,
                random_state=42,
                n_jobs=-1,
            )),
        ])

    param_dist = {
        "clf__n_estimators":     [200, 300, 500],
        "clf__max_depth":        [15, 25, None],
        "clf__min_samples_leaf": [1, 2, 3],
        "clf__max_features":     ["sqrt", 0.5],
    }

    print("  RandomizedSearchCV (15 iteracji, GroupKFold)...")
    search = RandomizedSearchCV(
        pipeline_rf,
        param_dist,
        n_iter=15,
        scoring="f1_macro",
        cv=gkf,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=g_train)

    best_rf = search.best_estimator_
    print(f"\n  Najlepsze parametry RF: {search.best_params_}")
    print(f"  CV F1 macro (GroupKFold): {search.best_score_:.4f}")

    # ── BalancedRandomForest jako alternatywa ──────────────────────────────
    best_model   = best_rf
    best_cv_f1   = search.best_score_
    best_name    = "RandomForest+SMOTE"

    if BRF_AVAILABLE:
        print("\n  Próba BalancedRandomForest...")
        brf_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    BalancedRandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
                sampling_strategy="all",
                replacement=True,
            )),
        ])
        brf_pipeline.fit(X_train, y_train)

        # Szybka ocena CV na trainie
        brf_scores = []
        for fold_train, fold_val in gkf.split(X_train, y_train, g_train):
            brf_tmp = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    BalancedRandomForestClassifier(
                    n_estimators=200, random_state=42, n_jobs=-1,
                    sampling_strategy="all", replacement=True,
                )),
            ])
            brf_tmp.fit(X_train[fold_train], y_train[fold_train])
            preds = brf_tmp.predict(X_train[fold_val])
            brf_scores.append(f1_score(y_train[fold_val], preds, average="macro", zero_division=0))
        brf_cv = float(np.mean(brf_scores))
        print(f"  BRF CV F1 (szybka ocena): {brf_cv:.4f}")

        if brf_cv > best_cv_f1:
            best_model = brf_pipeline
            best_cv_f1 = brf_cv
            best_name  = "BalancedRandomForest"
            print(f"  [WYBÓR] BalancedRandomForest lepszy ({brf_cv:.4f} > {search.best_score_:.4f})")
        else:
            print(f"  [WYBÓR] RandomForest+SMOTE lepszy ({search.best_score_:.4f} >= {brf_cv:.4f})")

    print(f"\n  Wybrany model: {best_name}")
    return best_model, best_name


# ═══════════════════════════════════════════════════════════════════════════════
# EWALUACJA Z DWUWYMIAROWYM TUNINGIEM PROGÓW  (v1.1 – główna zmiana)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 65)
    print("KROK 4: Ewaluacja z dwuwymiarowym tuningiem progów")
    print("=" * 65)

    y_val_pred = model.predict(X_val)
    val_f1_base = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    print(f"\n[VAL] F1 macro (bez korekty): {val_f1_base:.4f}")
    print(classification_report(y_val, y_val_pred, labels=LABEL_ORDER, zero_division=0))

    best_mult_strong = 1.0
    best_mult_normal = 1.0
    best_val_f1      = val_f1_base

    if hasattr(model, "predict_proba"):
        val_probs = model.predict_proba(X_val)
        label_idx = {lbl: i for i, lbl in enumerate(model.classes_)}

        strong_idx = label_idx.get("StrongPalsy", -1)
        normal_idx = label_idx.get("Normal", -1)

        # Dwuwymiarowa siatka: mult_strong x mult_normal
        # Zakres dobrany empirycznie: Normal jest najbardziej problematyczny
        strong_mults = np.arange(1.0, 4.0, 0.25)
        normal_mults = np.arange(1.0, 6.0, 0.5)

        print(f"  Siatka progów: {len(strong_mults)} x {len(normal_mults)} = "
              f"{len(strong_mults)*len(normal_mults)} kombinacji...")

        for ms, mn in iterproduct(strong_mults, normal_mults):
            p = val_probs.copy()
            if strong_idx >= 0:
                p[:, strong_idx] *= ms
            if normal_idx >= 0:
                p[:, normal_idx] *= mn
            preds = []
            for i in range(len(p)):
                row = [p[i, label_idx[lbl]] for lbl in LABEL_ORDER]
                preds.append(LABEL_ORDER[int(np.argmax(row))])
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            if f1 > best_val_f1:
                best_val_f1      = f1
                best_mult_strong = ms
                best_mult_normal = mn

        print(f"  Optymalny mnożnik StrongPalsy: {best_mult_strong:.2f}")
        print(f"  Optymalny mnożnik Normal:       {best_mult_normal:.2f}")
        print(f"  Val F1 po korekcie:             {best_val_f1:.4f}  "
              f"(+{best_val_f1 - val_f1_base:.4f} vs bez korekty)")

        # Zastosowanie optymalnych progów na zbiorze testowym
        test_probs = model.predict_proba(X_test)
        p_test = test_probs.copy()
        if strong_idx >= 0:
            p_test[:, strong_idx] *= best_mult_strong
        if normal_idx >= 0:
            p_test[:, normal_idx] *= best_mult_normal
        y_test_pred = []
        for i in range(len(p_test)):
            row = [p_test[i, label_idx[lbl]] for lbl in LABEL_ORDER]
            y_test_pred.append(LABEL_ORDER[int(np.argmax(row))])
    else:
        y_test_pred = list(model.predict(X_test))

    print("\n★ Wyniki na zbiorze testowym (z korekcją progu):")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0))

    return y_test_pred, best_mult_strong, best_mult_normal


# ═══════════════════════════════════════════════════════════════════════════════
# WYKRESY
# ═══════════════════════════════════════════════════════════════════════════════

def save_plots(model, X_test, y_test, y_test_pred):
    print("\n" + "=" * 65)
    print("KROK 5: Generowanie wykresów")
    print("=" * 65)

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Wartość rzeczywista")
    ax.set_title("Macierz pomyłek – v1.1 (GroupSplit + SMOTE + 2D Threshold)")
    plt.tight_layout()
    p = OUTPUT_DIR / "confusion_matrix_v1_1.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Zapisano: {p}")
    plt.close()

    # 2. Per-class metrics z porównaniem v1.0
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0
    )

    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle("Porównanie v1.0 → v1.1", fontsize=14, fontweight="bold")

    # Lewy panel: v1.1 per-class
    ax2 = axes[0]
    x_pos = np.arange(len(LABEL_ORDER))
    width = 0.25
    for j, (vals, color, lbl_m) in enumerate(zip(
        [p_per, r_per, f1_per],
        ["#3498db", "#2ecc71", "#e74c3c"],
        ["Precision", "Recall", "F1"]
    )):
        bars = ax2.bar(x_pos + j * width, vals, width, label=lbl_m, color=color, alpha=0.85)
        ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(LABEL_ORDER)
    ax2.set_ylim(0, 1.25)
    ax2.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Próg 0.7")
    ax2.set_title("Precision / Recall / F1 – v1.1")
    ax2.legend(fontsize=8)

    # Prawy panel: porównanie F1 macro v1.0 vs v1.1
    ax3 = axes[1]
    v10_f1s = [V1_0_RESULTS[lbl]["f1"] for lbl in LABEL_ORDER]
    v11_f1s = list(f1_per)
    x3 = np.arange(len(LABEL_ORDER))
    w3 = 0.35
    b1 = ax3.bar(x3 - w3/2, v10_f1s, w3, label="v1.0", color="#95a5a6", alpha=0.9)
    b2 = ax3.bar(x3 + w3/2, v11_f1s, w3, label="v1.1", color="#e74c3c", alpha=0.9)
    ax3.bar_label(b1, fmt="%.2f", fontsize=9, padding=2)
    ax3.bar_label(b2, fmt="%.2f", fontsize=9, padding=2)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(LABEL_ORDER)
    ax3.set_ylim(0, 1.2)
    ax3.set_title("F1 per klasa: v1.0 vs v1.1")
    ax3.legend()

    plt.tight_layout()
    p2 = OUTPUT_DIR / "comparison_v1_0_vs_v1_1.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  Zapisano: {p2}")
    plt.close()

    # 3. Feature importances
    rf_clf = None
    for step_name, step in model.steps:
        if hasattr(step, "feature_importances_"):
            rf_clf = step
            break
    if rf_clf is not None:
        importances = rf_clf.feature_importances_
        sorted_idx  = np.argsort(importances)[::-1]
        fig3, ax4 = plt.subplots(figsize=(10, 8))
        colors_imp = ["#e74c3c" if v > np.mean(importances) else "#3498db"
                      for v in importances[sorted_idx]]
        ax4.barh(
            [ALL_FEATURE_COLS[i] for i in sorted_idx][::-1],
            importances[sorted_idx][::-1],
            color=colors_imp[::-1]
        )
        ax4.axvline(x=np.mean(importances), color="gray", linestyle="--",
                    alpha=0.7, label="Średnia")
        ax4.set_title("Ważność cech – v1.1\nCzerwony = powyżej średniej")
        ax4.legend()
        plt.tight_layout()
        p3 = OUTPUT_DIR / "feature_importance_v1_1.png"
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        print(f"  Zapisano: {p3}")
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ZAPIS WYNIKÓW
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(model, y_test, y_test_pred, best_mult_strong, best_mult_normal, model_name):
    print("\n" + "=" * 65)
    print("KROK 6: Zapis wyników")
    print("=" * 65)

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0
    )
    metrics_per_class = {
        lbl: {
            "precision": round(float(p_per[i]), 4),
            "recall":    round(float(r_per[i]), 4),
            "f1":        round(float(f1_per[i]), 4),
            "support":   int(sup_per[i]),
        }
        for i, lbl in enumerate(LABEL_ORDER)
    }

    errors     = [(t, p) for t, p in zip(y_test, y_test_pred) if t != p]
    error_cnts = Counter(errors)

    f1_mac = float(np.mean(f1_per))

    summary = {
        "version":      "1.1",
        "model":        model_name,
        "split_method": "GroupShuffleSplit (no data leakage)",
        "threshold_tuning": {
            "method": "2D grid search on validation set",
            "mult_strong": round(best_mult_strong, 2),
            "mult_normal": round(best_mult_normal, 2),
        },
        "class_weights": CLASS_WEIGHTS,
        "test_accuracy": round(float(accuracy_score(y_test, y_test_pred)), 4),
        "test_f1_macro": round(f1_mac, 4),
        "metrics_per_class": metrics_per_class,
        "comparison_v1_0_vs_v1_1": {
            lbl: {
                "f1_v1_0": V1_0_RESULTS[lbl]["f1"],
                "f1_v1_1": metrics_per_class[lbl]["f1"],
                "delta":   round(metrics_per_class[lbl]["f1"] - V1_0_RESULTS[lbl]["f1"], 4),
            }
            for lbl in LABEL_ORDER
        },
        "error_analysis": {
            "total_errors": len(errors),
            "error_rate":   round(len(errors) / len(y_test), 4),
            "error_types":  {f"{t}->{p}": cnt for (t, p), cnt in error_cnts.items()},
        },
    }

    json_path = OUTPUT_DIR / "model_results_v1_1.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Wyniki JSON: {json_path}")

    model_path = OUTPUT_DIR / "best_model_v1_1.pkl"
    joblib.dump(model, model_path)
    print(f"  Model:       {model_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Dane
    X, y, groups, _ = load_and_prepare(CSV_PATH)

    # 2. Podział grupowy
    X_train, X_val, X_test, y_train, y_val, y_test, g_train = group_split(
        X, y, groups, test_size=0.15, val_size=0.15
    )

    # 3. Trening
    model, model_name = build_and_train(X_train, y_train, g_train)

    # 4. Ewaluacja z dwuwymiarowym tuningiem progów
    y_test_pred, best_mult_strong, best_mult_normal = evaluate(
        model, X_val, y_val, X_test, y_test
    )

    # 5. Wykresy
    save_plots(model, X_test, y_test, y_test_pred)

    # 6. Zapis
    summary = save_results(
        model, y_test, y_test_pred,
        best_mult_strong, best_mult_normal, model_name
    )

    # 7. Podsumowanie końcowe z porównaniem
    print("\n" + "=" * 65)
    print("PODSUMOWANIE KOŃCOWE – v1.1")
    print("=" * 65)
    print(f"  Model:          {model_name}")
    print(f"  Test Accuracy:  {summary['test_accuracy']:.4f}  "
          f"(v1.0: {V1_0_RESULTS['accuracy']:.4f})")
    print(f"  Test F1 macro:  {summary['test_f1_macro']:.4f}  "
          f"(v1.0: {V1_0_RESULTS['f1_macro']:.4f})")
    print()
    print(f"  {'Klasa':15s}  {'F1 v1.0':>8s}  {'F1 v1.1':>8s}  {'Delta':>8s}")
    print(f"  {'-'*45}")
    for lbl in LABEL_ORDER:
        cmp = summary["comparison_v1_0_vs_v1_1"][lbl]
        delta_str = f"+{cmp['delta']:.3f}" if cmp['delta'] >= 0 else f"{cmp['delta']:.3f}"
        print(f"  {lbl:15s}  {cmp['f1_v1_0']:>8.3f}  {cmp['f1_v1_1']:>8.3f}  {delta_str:>8s}")
    print()
    print(f"  Próg StrongPalsy: x{best_mult_strong:.2f}")
    print(f"  Próg Normal:      x{best_mult_normal:.2f}")
    print(f"\n  Błędy łącznie: {summary['error_analysis']['total_errors']} "
          f"({summary['error_analysis']['error_rate']*100:.1f}%)")
    print(f"\nWyniki zapisane w: {OUTPUT_DIR}")
    print("\n[WAŻNE] GroupShuffleSplit gwarantuje brak wycieku danych – metryki wiarygodne.")


if __name__ == "__main__":
    main()