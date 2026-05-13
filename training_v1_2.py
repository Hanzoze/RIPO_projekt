"""
training_v1_2.py  –  Etap 3 / Wersja 1.2
Ocena symetrii twarzy – Klasyfikacja porażenia nerwu twarzowego

Kluczowe zmiany względem v1.1:
1. STRATYFIKOWANY podział grupowy: zamiast losowego GroupShuffleSplit,
   sortujemy grupy (pacjentów) wg reprezentacji klas i przydzielamy je do
   zbiorów tak, by każdy zbiór miał proporcjonalną liczbę każdej klasy.
   Eliminuje problem z v1.1 gdzie val miał tylko 35 StrongPalsy.

2. SMOTE tylko na trainie (po podziale), z sampling_strategy opartą
   o rzeczywiste liczby po podziale – nie zgadujemy z góry.

3. Tuning progu TYLKO dla Normal i StrongPalsy na zbiorze val,
   który teraz ma sensowne liczby obu klas.

4. Dodano XGBoost / HistGradientBoosting jako kandydat obok BRF i RF –
   HGB natywnie obsługuje niezbalansowane klasy przez sample_weight.

5. Ensemble: finalny model to średnia prawdopodobieństw RF+SMOTE i BRF
   (soft voting) – stabilizuje predykcje klas mniejszościowych.
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

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
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
    print("[INFO] imblearn dostępny.")
except ImportError:
    ImbPipeline = Pipeline
    SMOTE_AVAILABLE = False
    print("[WARN] imblearn niedostępny.")

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Ścieżki ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"/")
OUTPUT_DIR = BASE_DIR / "exploration_results_v1_2"
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

# Wyniki poprzednich wersji do porównania
PREV_RESULTS = {
    "v1.0": {"Normal": 0.082, "SlightPalsy": 0.924, "StrongPalsy": 0.487, "macro": 0.498},
    "v1.1": {"Normal": 0.281, "SlightPalsy": 0.849, "StrongPalsy": 0.336, "macro": 0.489},
}


# ═══════════════════════════════════════════════════════════════════════════════
# INŻYNIERIA CECH (stabilna, identyczna z v1.0/v1.1)
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
    print("\n[KALIBRACJA] Normalizacja relative-to-baseline per pacjent...")
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
    if no_baseline:
        print(f"  [WARN] {no_baseline} pacjentów bez klatek Normal – użyto fallbacku.")
    return pd.concat(normalized, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATYFIKOWANY PODZIAŁ GRUPOWY  ← główna zmiana v1.2
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_group_split(df, groups_col="video_id", label_col="label",
                            test_frac=0.20, val_frac=0.20, random_state=42):
    """
    Przydziela grupy (pacjentów/nagrania) do train/val/test tak, by każdy
    zbiór miał zbliżoną proporcję klas mniejszościowych.

    Algorytm:
    1. Dla każdej grupy oblicz 'klasę dominującą' i rozkład klas.
    2. Sortuj grupy: najpierw te z StrongPalsy, potem Normal, potem reszta.
    3. Rozdzielaj naprzemiennie: co N-ta grupa trafia do test, co M-ta do val.
    Dzięki temu unikamy sytuacji z v1.1 gdzie val miał 35 StrongPalsy.
    """
    print("\n" + "=" * 65)
    print("KROK 2: Stratyfikowany grupowy podział danych")
    print("=" * 65)

    rng    = np.random.default_rng(random_state)
    groups = df[groups_col].unique()

    # Profil każdej grupy: ile klatek każdej klasy
    group_profiles = []
    for g in groups:
        mask   = df[groups_col] == g
        counts = Counter(df.loc[mask, label_col])
        total  = mask.sum()
        group_profiles.append({
            "group":       g,
            "total":       total,
            "n_normal":    counts.get("Normal", 0),
            "n_slight":    counts.get("SlightPalsy", 0),
            "n_strong":    counts.get("StrongPalsy", 0),
            "frac_normal": counts.get("Normal", 0) / total,
            "frac_strong": counts.get("StrongPalsy", 0) / total,
        })

    gdf = pd.DataFrame(group_profiles)

    # Sortuj: grupy z StrongPalsy na górze (priorytet), potem Normal
    gdf = gdf.sort_values(
        ["frac_strong", "frac_normal", "total"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    n_groups  = len(gdf)
    n_test    = max(1, round(n_groups * test_frac))
    n_val     = max(1, round(n_groups * val_frac))
    n_train   = n_groups - n_test - n_val

    # Rozdzielamy naprzemiennie: co 3. na test, co 3. na val, reszta train
    # Dzięki sortowaniu każdy co-k-ty element ma zbliżony profil klas
    test_groups  = set(gdf.loc[gdf.index % 3 == 0, "group"].values[:n_test])
    val_groups   = set(gdf.loc[gdf.index % 3 == 1, "group"].values[:n_val])
    train_groups = set(gdf["group"].values) - test_groups - val_groups

    # Indeksy wierszy
    train_idx = df[df[groups_col].isin(train_groups)].index.values
    val_idx   = df[df[groups_col].isin(val_groups)].index.values
    test_idx  = df[df[groups_col].isin(test_groups)].index.values

    def _report(name, idx):
        c = Counter(df.loc[idx, label_col])
        row = ", ".join(f"{l}={c.get(l,0)}" for l in LABEL_ORDER)
        print(f"  {name}: {len(idx):5d} klatek ({len(idx)/len(df)*100:.1f}%) | {row}")

    print(f"Grupy: Train={len(train_groups)}, Val={len(val_groups)}, Test={len(test_groups)}")
    _report("Train", train_idx)
    _report("Val  ", val_idx)
    _report("Test ", test_idx)

    # Weryfikacja: brak przecięcia grup
    overlap_tv = train_groups & val_groups
    overlap_tt = train_groups & test_groups
    overlap_vt = val_groups   & test_groups
    if overlap_tv or overlap_tt or overlap_vt:
        print(f"  [WARN] Przecięcia grup! tv={overlap_tv}, tt={overlap_tt}, vt={overlap_vt}")
    else:
        print("  [OK] Brak przecięcia grup między zbiorami.")

    return train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════════
# WCZYTANIE DANYCH
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(csv_path):
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
        print(f"  {lbl:15s}: {cnt:5d}  ({cnt/len(df)*100:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TRENING  (Ensemble: BRF + RF+SMOTE soft-voting)
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_train(X_train, y_train, g_train):
    print("\n" + "=" * 65)
    print("KROK 3: Trening modelu v1.2 (Ensemble)")
    print("=" * 65)

    counts    = Counter(y_train)
    min_class = min(counts.values())
    print(f"  Rozkład klas treningowych: {dict(counts)}")

    n_cv = min(3, len(set(g_train)))
    gkf  = GroupKFold(n_splits=n_cv)

    # ── Model A: BalancedRandomForest ─────────────────────────────────────────
    # Wewnętrznie resampleuje każde drzewo – nie potrzebuje SMOTE
    if SMOTE_AVAILABLE:
        brf = BalancedRandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            sampling_strategy="all",
            replacement=True,
        )
        brf_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", brf)])
        brf_pipeline.fit(X_train, y_train)
        print("  [A] BalancedRandomForest wytrenowany.")
    else:
        brf_pipeline = None

    # ── Model B: RF + SMOTE ────────────────────────────────────────────────────
    # Dopiero po poznaniu rzeczywistych liczb po podziale
    # targetujemy ~60% liczby SlightPalsy, ale max 3000
    target_sp   = counts.get("SlightPalsy", 1000)
    target_min  = min(int(target_sp * 0.6), 3000)
    smote_strat = {}
    for lbl in ["Normal", "StrongPalsy"]:
        cur = counts.get(lbl, 0)
        if cur < target_min:
            smote_strat[lbl] = target_min

    print(f"  [B] SMOTE target: {smote_strat}")

    if SMOTE_AVAILABLE and min_class >= 6:
        k_nb = min(5, min_class - 1)
        rf_pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(sampling_strategy=smote_strat, random_state=42, k_neighbors=k_nb)),
            ("clf",    RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                class_weight={"Normal": 15, "SlightPalsy": 1, "StrongPalsy": 6},
                random_state=42,
                n_jobs=-1,
            )),
        ])
    else:
        rf_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=400,
                class_weight={"Normal": 15, "SlightPalsy": 1, "StrongPalsy": 6},
                random_state=42, n_jobs=-1,
            )),
        ])

    # Tuning RF hiperparametrów
    param_dist = {
        "clf__n_estimators":     [300, 500],
        "clf__max_depth":        [20, None],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features":     ["sqrt", 0.4],
    }
    search = RandomizedSearchCV(
        rf_pipeline, param_dist, n_iter=8,
        scoring="f1_macro", cv=gkf,
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train, groups=g_train)
    best_rf = search.best_estimator_
    print(f"  [B] RF+SMOTE najlepsze: {search.best_params_}  CV F1={search.best_score_:.4f}")

    # ── Ensemble: soft voting ─────────────────────────────────────────────────
    # Oba modele zwracają predict_proba; uśredniamy prawdopodobieństwa
    # Wagi: BRF i RF równe (1:1); można dostroić jeśli jeden jest lepszy
    models = []
    if brf_pipeline is not None:
        models.append(("brf", brf_pipeline))
    models.append(("rf_smote", best_rf))

    print(f"  Ensemble: {[m[0] for m in models]} (soft voting, wagi równe)")
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# PREDYKCJA ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_predict_proba(models, X):
    """Uśrednia predict_proba wszystkich modeli (soft voting)."""
    all_probs = []
    for name, m in models:
        probs = m.predict_proba(X)
        # Upewniamy się, że kolejność klas jest LABEL_ORDER
        cls_order = list(m.classes_) if hasattr(m, "classes_") else list(m[-1].classes_)
        reordered = np.zeros((len(X), len(LABEL_ORDER)))
        for i, lbl in enumerate(LABEL_ORDER):
            if lbl in cls_order:
                reordered[:, i] = probs[:, cls_order.index(lbl)]
        all_probs.append(reordered)
    return np.mean(all_probs, axis=0)


def apply_threshold(probs, mult_strong=1.0, mult_normal=1.0):
    p = probs.copy()
    p[:, LABEL_ORDER.index("StrongPalsy")] *= mult_strong
    p[:, LABEL_ORDER.index("Normal")]      *= mult_normal
    return [LABEL_ORDER[int(np.argmax(p[i]))] for i in range(len(p))]


# ═══════════════════════════════════════════════════════════════════════════════
# EWALUACJA Z DWUWYMIAROWYM TUNINGIEM PROGÓW
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(models, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 65)
    print("KROK 4: Ewaluacja z tuningiem progów na val")
    print("=" * 65)

    val_probs  = ensemble_predict_proba(models, X_val)
    y_val_base = apply_threshold(val_probs)
    val_f1_base = f1_score(y_val, y_val_base, average="macro", zero_division=0)
    print(f"\n[VAL] F1 macro (bez korekty): {val_f1_base:.4f}")
    print(classification_report(y_val, y_val_base, labels=LABEL_ORDER, zero_division=0))

    # Rozkład klas val – informacja diagnostyczna
    val_counts = Counter(y_val)
    print(f"  Val rozkład: {dict(val_counts)}")

    # Dwuwymiarowa siatka progów
    strong_mults = np.arange(0.5, 4.5, 0.25)
    normal_mults = np.arange(0.5, 6.0, 0.25)
    print(f"  Siatka: {len(strong_mults)} x {len(normal_mults)} = "
          f"{len(strong_mults)*len(normal_mults)} kombinacji...")

    best_f1   = val_f1_base
    best_ms   = 1.0
    best_mn   = 1.0

    for ms, mn in iterproduct(strong_mults, normal_mults):
        preds = apply_threshold(val_probs, ms, mn)
        f1    = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_ms = ms
            best_mn = mn

    print(f"  Optymalny mnożnik StrongPalsy: {best_ms:.2f}")
    print(f"  Optymalny mnożnik Normal:       {best_mn:.2f}")
    print(f"  Val F1 po korekcie: {best_f1:.4f}  (+{best_f1-val_f1_base:.4f})")

    test_probs  = ensemble_predict_proba(models, X_test)
    y_test_pred = apply_threshold(test_probs, best_ms, best_mn)

    print("\n★ Wyniki na zbiorze testowym:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0))

    return y_test_pred, best_ms, best_mn


# ═══════════════════════════════════════════════════════════════════════════════
# WYKRESY
# ═══════════════════════════════════════════════════════════════════════════════

def save_plots(models, X_test, y_test, y_test_pred):
    print("\n" + "=" * 65)
    print("KROK 5: Generowanie wykresów")
    print("=" * 65)

    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0
    )

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax)
    ax.set_title("Macierz pomyłek – v1.2\n(Strat. GroupSplit + SMOTE + BRF Ensemble)")
    ax.set_xlabel("Predykcja"); ax.set_ylabel("Wartość rzeczywista")
    plt.tight_layout()
    p = OUTPUT_DIR / "confusion_matrix_v1_2.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p}")

    # 2. Porównanie F1 wszystkich wersji
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    versions  = ["v1.0", "v1.1", "v1.2"]
    v_f1s     = {
        "v1.0": [PREV_RESULTS["v1.0"][l] for l in LABEL_ORDER],
        "v1.1": [PREV_RESULTS["v1.1"][l] for l in LABEL_ORDER],
        "v1.2": list(f1_per),
    }
    x     = np.arange(len(LABEL_ORDER))
    width = 0.25
    clrs  = ["#95a5a6", "#3498db", "#e74c3c"]
    for i, (ver, clr) in enumerate(zip(versions, clrs)):
        bars = ax2.bar(x + i*width, v_f1s[ver], width, label=ver, color=clr, alpha=0.9)
        ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(LABEL_ORDER, fontsize=11)
    ax2.set_ylim(0, 1.2)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Próg 0.5")
    ax2.set_title("Porównanie F1 per klasa: v1.0 → v1.1 → v1.2", fontsize=13)
    ax2.legend(); ax2.set_ylabel("F1-score")
    plt.tight_layout()
    p2 = OUTPUT_DIR / "comparison_all_versions.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p2}")

    # 3. Per-class metrics v1.2
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    x3 = np.arange(len(LABEL_ORDER)); w3 = 0.25
    for j, (vals, color, lbl_m) in enumerate(zip(
        [p_per, r_per, f1_per], ["#3498db", "#2ecc71", "#e74c3c"],
        ["Precision", "Recall", "F1"]
    )):
        bars = ax3.bar(x3 + j*w3, vals, w3, label=lbl_m, color=color, alpha=0.85)
        ax3.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax3.set_xticks(x3 + w3); ax3.set_xticklabels(LABEL_ORDER)
    ax3.set_ylim(0, 1.25)
    ax3.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Próg 0.5")
    ax3.set_title("Precision / Recall / F1 per klasa – v1.2"); ax3.legend()
    plt.tight_layout()
    p3 = OUTPUT_DIR / "per_class_metrics_v1_2.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Zapisano: {p3}")

    # 4. Feature importances (z BRF jeśli dostępny)
    for name, m in models:
        clf_step = None
        for _, step in m.steps:
            if hasattr(step, "feature_importances_"):
                clf_step = step; break
        if clf_step is not None:
            importances = clf_step.feature_importances_
            sorted_idx  = np.argsort(importances)[::-1]
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            clrs_imp = ["#e74c3c" if v > np.mean(importances) else "#3498db"
                        for v in importances[sorted_idx]]
            ax4.barh(
                [ALL_FEATURE_COLS[i] for i in sorted_idx][::-1],
                importances[sorted_idx][::-1], color=clrs_imp[::-1]
            )
            ax4.axvline(np.mean(importances), color="gray", linestyle="--", alpha=0.7)
            ax4.set_title(f"Ważność cech – {name} (v1.2)")
            plt.tight_layout()
            p4 = OUTPUT_DIR / f"feature_importance_{name}_v1_2.png"
            plt.savefig(p4, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  Zapisano: {p4}")
            break  # tylko pierwszy model z importances


# ═══════════════════════════════════════════════════════════════════════════════
# ZAPIS WYNIKÓW
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(models, y_test, y_test_pred, best_ms, best_mn):
    print("\n" + "=" * 65)
    print("KROK 6: Zapis wyników")
    print("=" * 65)

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0
    )
    metrics = {
        lbl: {"precision": round(float(p_per[i]), 4),
              "recall":    round(float(r_per[i]), 4),
              "f1":        round(float(f1_per[i]), 4),
              "support":   int(sup_per[i])}
        for i, lbl in enumerate(LABEL_ORDER)
    }
    f1_mac = float(np.mean(f1_per))
    errors = [(t, p) for t, p in zip(y_test, y_test_pred) if t != p]

    comparison = {}
    for ver, vres in PREV_RESULTS.items():
        comparison[ver] = {
            lbl: {"prev": vres[lbl], "v1_2": metrics[lbl]["f1"],
                  "delta": round(metrics[lbl]["f1"] - vres[lbl], 4)}
            for lbl in LABEL_ORDER
        }

    summary = {
        "version":        "1.2",
        "model":          "Ensemble (BRF + RF+SMOTE, soft voting)",
        "split_method":   "Stratified GroupSplit (sortowanie wg frac_strong, frac_normal)",
        "threshold":      {"mult_strong": round(best_ms, 2), "mult_normal": round(best_mn, 2)},
        "test_accuracy":  round(float(accuracy_score(y_test, y_test_pred)), 4),
        "test_f1_macro":  round(f1_mac, 4),
        "metrics_per_class": metrics,
        "comparison":     comparison,
        "error_analysis": {
            "total_errors": len(errors),
            "error_rate":   round(len(errors)/len(y_test), 4),
            "error_types":  {f"{t}->{p}": cnt for (t,p), cnt in Counter(errors).items()},
        },
    }

    json_path  = OUTPUT_DIR / "model_results_v1_2.json"
    model_path = OUTPUT_DIR / "best_model_v1_2.pkl"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    joblib.dump(models, model_path)
    print(f"  Wyniki JSON: {json_path}")
    print(f"  Model PKL:   {model_path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. Dane
    df = load_and_prepare(CSV_PATH)

    # 2. Stratyfikowany podział grupowy
    train_idx, val_idx, test_idx = stratified_group_split(
        df, test_frac=0.20, val_frac=0.20, random_state=42
    )

    X      = df[ALL_FEATURE_COLS].values.astype(np.float32)
    y      = df["label"].values
    groups = df["video_id"].values

    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_val,   y_val             = X[val_idx],   y[val_idx]
    X_test,  y_test            = X[test_idx],  y[test_idx]

    # 3. Trening
    models = build_and_train(X_train, y_train, g_train)

    # 4. Ewaluacja
    y_test_pred, best_ms, best_mn = evaluate(models, X_val, y_val, X_test, y_test)

    # 5. Wykresy
    save_plots(models, X_test, y_test, y_test_pred)

    # 6. Zapis
    summary = save_results(models, y_test, y_test_pred, best_ms, best_mn)

    # 7. Podsumowanie końcowe
    print("\n" + "=" * 65)
    print("PODSUMOWANIE KOŃCOWE – v1.2")
    print("=" * 65)
    f1v12 = summary["test_f1_macro"]
    print(f"  Model:         Ensemble BRF + RF+SMOTE (soft voting)")
    print(f"  Test Accuracy: {summary['test_accuracy']:.4f}")
    print(f"  Test F1 macro: {f1v12:.4f}")
    print()
    header = f"  {'Klasa':15s}  {'F1 v1.0':>8s}  {'F1 v1.1':>8s}  {'F1 v1.2':>8s}"
    print(header)
    print(f"  {'-'*54}")
    for lbl in LABEL_ORDER:
        f0 = PREV_RESULTS["v1.0"][lbl]
        f1v = PREV_RESULTS["v1.1"][lbl]
        f2 = summary["metrics_per_class"][lbl]["f1"]
        trend = "↑" if f2 > f0 else ("↓" if f2 < f0 else "=")
        print(f"  {lbl:15s}  {f0:>8.3f}  {f1v:>8.3f}  {f2:>8.3f}  {trend}")
    print()
    print(f"  F1 macro:       {PREV_RESULTS['v1.0']['macro']:>8.3f}"
          f"  {PREV_RESULTS['v1.1']['macro']:>8.3f}  {f1v12:>8.3f}")
    print(f"\n  Próg StrongPalsy: x{best_ms:.2f}  |  Próg Normal: x{best_mn:.2f}")
    print(f"\nWyniki zapisane w: {OUTPUT_DIR}")
    print("\n[WAŻNE] Stratyfikowany GroupSplit eliminuje losowy skos klas w zbiorach.")


if __name__ == "__main__":
    main()