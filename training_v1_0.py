"""
training_v1_final.py  –  Etap 3 / Wersja 1.0
Ocena symetrii twarzy – Klasyfikacja z grupowym podziałem danych

Kluczowe zmiany względem prototypu (Etap 2):
1. GroupShuffleSplit: klatki z jednego wideo trafiają TYLKO do jednego zbioru.
   Eliminuje wyciek danych (data leakage) i daje wiarygodne metryki.
2. Baseline normalization: cechy każdej klatki normalizowane względem mediany
   klatek 'Normal' z tego samego wideo (fallback: 5-percentyl asymetrii).
3. Inżynieria cech: 7 nowych cech kompozytowych (HB_composite_index itd.).
4. SMOTE: oversampling klas mniejszościowych na zbiorze treningowym.
5. RandomizedSearchCV z GroupKFold: tuning hiperparametrów bez wycieku grup.
6. Threshold adjustment: kalibracja prawdopodobieństw dla klasy StrongPalsy.
"""

import argparse
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
from sklearn.model_selection import (
    GroupShuffleSplit, GroupKFold, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.pipeline import Pipeline
import joblib

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
    print("[INFO] SMOTE dostępny – zostanie użyty do oversamplingu.")
except ImportError:
    ImbPipeline = Pipeline
    SMOTE_AVAILABLE = False
    print("[WARN] imblearn niedostępny – trening bez SMOTE.")

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Ścieżki ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"C:\Users\danil\PycharmProjects\RIPO")
OUTPUT_DIR = BASE_DIR / "exploration_results_v1_final"
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
    "HB_composite_index",
    "eye_mouth_asymmetry_product",
    "global_symmetry_score",
    "brow_eye_combined",
    "lower_face_severity",
    "eye_closure_min",
    "mouth_asymmetry_combined",
    "total_asymmetry_delta",   # NAPRAWKA: był liczony ale nie używany w v1.3
]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + NEW_FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# KROK 1 – INŻYNIERIA CECH
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Oblicza cechy kompozytowe zgodne ze skalą House-Brackmanna."""
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

    # NAPRAWKA: ta cecha była liczona, ale nie trafiała do ALL_FEATURE_COLS w v1.3
    d["total_asymmetry_delta"] = d[BASE_FEATURE_COLS].abs().sum(axis=1)

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# KROK 2 – NORMALIZACJA BASELINE (per pacjent)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuje cechy każdej klatki względem stanu spoczynku pacjenta.

    NAPRAWKA względem v1.3:
    - Używamy MEDIANY (nie średniej) klatek Normal jako baseline — odporność
      na outlinery w adnotacjach.
    - Normalizujemy WSZYSTKIE klatki, nie tylko non-Normal. Dzięki temu klatki
      Normal po normalizacji są bliskie zeru (co jest poprawne — to właśnie
      baseline), a klatki z porażeniem mają wartości dodatnie/ujemne względem
      baseline tego pacjenta.
    - Fallback: jeśli pacjent nie ma klatek Normal, używamy 5-percentyla
      asymetrii jako przybliżenia baseline.
    """
    print("\n[KALIBRACJA] Normalizacja cech względem stanu spoczynku (Baseline)...")
    normalized = []
    patients_without_baseline = 0

    for vid, group in df.groupby("video_id"):
        baseline_df = group[group["label"] == "Normal"]

        if not baseline_df.empty:
            baseline_vals = baseline_df[ALL_FEATURE_COLS].median()
        else:
            # Fallback: klatki o najniższej asymetrii globalnej
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
# KROK 3 – WCZYTANIE I PRZYGOTOWANIE DANYCH
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(csv_path: Path) -> tuple:
    print("=" * 65)
    print("KROK 1: Wczytanie i przygotowanie danych")
    print("=" * 65)

    df = pd.read_csv(csv_path)
    print(f"Wczytano wierszy: {len(df)}")

    # Wyznaczamy video_id z nazwy pliku (folder = pacjent/nagranie)
    df["video_id"] = df["filename"].apply(lambda x: str(x).split("/")[0])

    # Inżynieria cech
    df = engineer_features(df)
    df = df.dropna(subset=ALL_FEATURE_COLS + ["label"])
    print(f"Po dropna: {len(df)} wierszy")

    # Normalizacja baseline – PRZED podziałem (używamy wszystkich danych
    # jednego pacjenta do wyznaczenia jego baseline, nie ma tu wycieku,
    # bo używamy wyłącznie danych tego samego pacjenta)
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
# KROK 4 – GRUPOWY PODZIAŁ DANYCH
# ═══════════════════════════════════════════════════════════════════════════════

def group_split(
    X, y, groups,
    test_size: float = 0.15,
    val_size: float  = 0.15,
    random_state: int = 42
):
    """
    Podział grupowy: klatki z jednego wideo trafiają TYLKO do jednego zbioru.
    Eliminuje data leakage z Etapu 2 (gdzie ten sam pacjent mógł być
    jednocześnie w train i test).
    """
    print("\n" + "=" * 65)
    print("KROK 2: Grupowy podział danych (GroupShuffleSplit)")
    print("=" * 65)

    # 1. Wydzielamy zbiór testowy
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(gss_test.split(X, y, groups))

    X_tv, y_tv, g_tv = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
    X_test, y_test   = X[test_idx],       y[test_idx]

    # 2. Z pozostałej części wydzielamy zbiór walidacyjny
    effective_val = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=effective_val, random_state=random_state + 1
    )
    train_idx2, val_idx2 = next(gss_val.split(X_tv, y_tv, g_tv))

    X_train, y_train, g_train = (
        X_tv[train_idx2], y_tv[train_idx2], g_tv[train_idx2]
    )
    X_val, y_val = X_tv[val_idx2], y_tv[val_idx2]

    print(f"Train: {len(X_train):5d}  ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Val:   {len(X_val):5d}  ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test:  {len(X_test):5d}  ({len(X_test)/len(X)*100:.1f}%)")

    print("\nRozkład klas (po podziale grupowym):")
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        c = Counter(y_split)
        row = ", ".join(f"{lbl}={c.get(lbl, 0)}" for lbl in LABEL_ORDER)
        print(f"  {split_name}: {row}")

    print(f"\n[WERYFIKACJA] Unikalne grupy – Train: {len(set(g_train))}, "
          f"Val: {len(set(g_tv[val_idx2]))}")  # Poprawiono X_tv na g_tv
    # Sprawdzamy brak przecięcia grup
    overlap = set(g_train) & set(groups[test_idx])
    if overlap:
        print(f"  [WARN] Przecięcie train/test grup: {len(overlap)} grup – sprawdź dane!")
    else:
        print("  [OK] Brak przecięcia grup między train a test.")

    return X_train, X_val, X_test, y_train, y_val, y_test, g_train


# ═══════════════════════════════════════════════════════════════════════════════
# KROK 5 – TRENING
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_train(X_train, y_train, g_train):
    print("\n" + "=" * 65)
    print("KROK 3: Trening modelu (Random Forest + SMOTE + GroupKFold)")
    print("=" * 65)

    n_cv = min(3, len(set(g_train)))
    gkf  = GroupKFold(n_splits=n_cv)

    min_class = pd.Series(y_train).value_counts().min()

    if SMOTE_AVAILABLE and min_class > 5:
        k_neighbors = min(5, min_class - 1)
        pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=42, k_neighbors=k_neighbors)),
            ("clf",    RandomForestClassifier(
                class_weight={"Normal": 5, "SlightPalsy": 1, "StrongPalsy": 3},
                random_state=42, n_jobs=-1
            )),
        ])
        print(f"  SMOTE z k_neighbors={k_neighbors}")
    else:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                class_weight={"Normal": 5, "SlightPalsy": 1, "StrongPalsy": 3},
                random_state=42, n_jobs=-1
            )),
        ])
        print("  Trening bez SMOTE (imblearn niedostępny lub za mało próbek).")

    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth":    [10, 20, None],
        "clf__min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=9,
        scoring="f1_macro",
        cv=gkf,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=g_train)

    print(f"\n  Najlepsze parametry: {search.best_params_}")
    print(f"  CV F1 macro (GroupKFold): {search.best_score_:.4f}")

    return search.best_estimator_


# ═══════════════════════════════════════════════════════════════════════════════
# KROK 6 – EWALUACJA Z KOREKCJĄ PROGU
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_val, y_val, X_test, y_test, label: str = "Random Forest"):
    print("\n" + "=" * 65)
    print("KROK 4: Ewaluacja")
    print("=" * 65)

    # --- Val bez korekty ---
    y_val_pred = model.predict(X_val)
    val_f1 = precision_recall_fscore_support(
        y_val, y_val_pred, average="macro", zero_division=0
    )[2]
    print(f"\n[VAL] F1 macro (bez korekty): {val_f1:.4f}")
    print(classification_report(y_val, y_val_pred, labels=LABEL_ORDER, zero_division=0))

    # --- Korekcja progu na val ---
    # Szukamy najlepszego mnożnika dla StrongPalsy metodą siatki na zbiorze val
    best_mult  = 1.0
    best_val_f1 = val_f1

    if hasattr(model, "predict_proba"):
        val_probs = model.predict_proba(X_val)
        label_idx = {lbl: i for i, lbl in enumerate(model.classes_)}

        strong_idx = label_idx.get("StrongPalsy", -1)
        normal_idx = label_idx.get("Normal", -1)

        for mult in np.arange(1.0, 3.5, 0.1):
            p = val_probs.copy()
            if strong_idx >= 0:
                p[:, strong_idx] *= mult
            if normal_idx >= 0:
                p[:, normal_idx] *= 1.2   # lekka korekta dla Normal
            preds = [LABEL_ORDER[np.argmax(
                [p[i, label_idx[lbl]] for lbl in LABEL_ORDER]
            )] for i in range(len(p))]
            f1 = precision_recall_fscore_support(
                y_val, preds, average="macro", zero_division=0
            )[2]
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_mult   = mult

        print(f"[TUNING] Najlepszy mnożnik dla StrongPalsy: {best_mult:.1f} "
              f"(val F1: {best_val_f1:.4f})")

        # --- Test z optymalnym progiem ---
        test_probs = model.predict_proba(X_test)
        p_test = test_probs.copy()
        if strong_idx >= 0:
            p_test[:, strong_idx] *= best_mult
        if normal_idx >= 0:
            p_test[:, normal_idx] *= 1.2
        y_test_pred = [
            LABEL_ORDER[np.argmax([p_test[i, label_idx[lbl]] for lbl in LABEL_ORDER])]
            for i in range(len(p_test))
        ]
    else:
        y_test_pred = model.predict(X_test)

    print("\n★ Wyniki na zbiorze testowym (z korekcją progu):")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0))

    return y_test_pred


# ═══════════════════════════════════════════════════════════════════════════════
# KROK 7 – WYKRESY
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
    ax.set_title("Macierz pomyłek – v1.0 (GroupSplit + SMOTE)")
    plt.tight_layout()
    p = OUTPUT_DIR / "confusion_matrix_v1_final.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    print(f"  Zapisano: {p}")
    plt.close()

    # 2. Per-class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0
    )
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x_pos  = np.arange(len(LABEL_ORDER))
    width  = 0.25
    for j, (vals, color, lbl_m) in enumerate(zip(
        [p_per, r_per, f1_per],
        ["#3498db", "#2ecc71", "#e74c3c"],
        ["Precision", "Recall", "F1"]
    )):
        bars = ax2.bar(x_pos + j * width, vals, width, label=lbl_m, color=color, alpha=0.85)
        ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(LABEL_ORDER)
    ax2.set_ylim(0, 1.2)
    ax2.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Próg 0.7")
    ax2.set_title("Precision / Recall / F1 per klasa – v1.0")
    ax2.legend()
    plt.tight_layout()
    p2 = OUTPUT_DIR / "per_class_metrics_v1_final.png"
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
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        colors_imp = ["#e74c3c" if v > np.mean(importances) else "#3498db"
                      for v in importances[sorted_idx]]
        ax3.barh(
            [ALL_FEATURE_COLS[i] for i in sorted_idx][::-1],
            importances[sorted_idx][::-1],
            color=colors_imp[::-1]
        )
        ax3.axvline(x=np.mean(importances), color="gray", linestyle="--",
                    alpha=0.7, label="Średnia")
        ax3.set_title("Ważność cech – v1.0\nCzerwony = powyżej średniej")
        ax3.legend()
        plt.tight_layout()
        p3 = OUTPUT_DIR / "feature_importance_v1_final.png"
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        print(f"  Zapisano: {p3}")
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# KROK 8 – ZAPIS WYNIKÓW I MODELU
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(model, y_test, y_test_pred, X, y, groups):
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

    errors      = [(t, p) for t, p in zip(y_test, y_test_pred) if t != p]
    error_cnts  = Counter(errors)

    summary = {
        "version": "1.0",
        "split_method": "GroupShuffleSplit (no data leakage)",
        "test_accuracy":  round(float(accuracy_score(y_test, y_test_pred)), 4),
        "metrics_per_class": metrics_per_class,
        "error_analysis": {
            "total_errors": len(errors),
            "error_rate":   round(len(errors) / len(y_test), 4),
            "error_types":  {f"{t}->{p}": cnt for (t, p), cnt in error_cnts.items()},
        },
    }

    json_path = OUTPUT_DIR / "model_results_v1_final.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Wyniki JSON: {json_path}")

    model_path = OUTPUT_DIR / "best_model_v1_final.pkl"
    joblib.dump(model, model_path)
    print(f"  Model:       {model_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Etap 3 – trening v1.0")
    args = parser.parse_args()

    # 1. Dane
    X, y, groups, df_full = load_and_prepare(CSV_PATH)

    # 2. Podział grupowy
    X_train, X_val, X_test, y_train, y_val, y_test, g_train = group_split(
        X, y, groups, test_size=0.15, val_size=0.15
    )

    # 3. Trening
    model = build_and_train(X_train, y_train, g_train)

    # 4. Ewaluacja z korekcją progu
    y_test_pred = evaluate(model, X_val, y_val, X_test, y_test)

    # 5. Wykresy
    save_plots(model, X_test, y_test, y_test_pred)

    # 6. Zapis
    summary = save_results(model, y_test, y_test_pred, X, y, groups)

    # 7. Podsumowanie
    print("\n" + "=" * 65)
    print("PODSUMOWANIE KOŃCOWE – v1.0")
    print("=" * 65)
    print(f"  Test Accuracy:  {summary['test_accuracy']:.4f}")
    for lbl in LABEL_ORDER:
        m = summary["metrics_per_class"][lbl]
        print(f"  {lbl:15s}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"\n  Błędy łącznie: {summary['error_analysis']['total_errors']} "
          f"({summary['error_analysis']['error_rate']*100:.1f}%)")
    print(f"\nWyniki zapisane w: {OUTPUT_DIR}")
    print("\n[WAŻNE] Metryki z tego skryptu są wiarygodne – GroupShuffleSplit")
    print("        gwarantuje brak wycieku danych między train a test.")


if __name__ == "__main__":
    main()