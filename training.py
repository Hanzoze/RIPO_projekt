import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import json
from pathlib import Path
from collections import Counter

from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics          import (classification_report, confusion_matrix,
                                       precision_recall_fscore_support, accuracy_score)
from sklearn.pipeline         import Pipeline
import joblib

matplotlib.use("Agg")  # bez GUI

# ============================================================
# KONFIGURACJA
# ============================================================
BASE_DIR    = Path(r"C:\Users\danil\PycharmProjects\RIPO")
OUTPUT_DIR  = BASE_DIR / "exploration_results"
CSV_PATH    = OUTPUT_DIR / "features.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

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

LABEL_ORDER = ["Normal", "SlightPalsy", "StrongPalsy"]

# ============================================================
# KROK 1: Wczytanie danych
# ============================================================
print("=" * 60)
print("KROK 1: Wczytanie danych")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"Wczytano wierszy: {len(df)}")
print(f"Kolumny: {list(df.columns)}")

# Usuń wiersze z brakującymi wartościami
df = df.dropna(subset=FEATURE_COLS + ["label"])
print(f"Po usunięciu NaN: {len(df)}")

# Rozkład klas
label_counts = Counter(df["label"])
print("\nRozkład klas:")
for lbl in LABEL_ORDER:
    cnt = label_counts.get(lbl, 0)
    print(f"  {lbl:15s}: {cnt:5d}  ({cnt/len(df)*100:.1f}%)")

X = df[FEATURE_COLS].values.astype(np.float32)
y = df["label"].values

# ============================================================
# KROK 2: Podział danych train/val/test
# ============================================================
print("\n" + "=" * 60)
print("KROK 2: Podział danych")
print("=" * 60)

# Najpierw wydziel test (15%), potem val (15% z pozostałych ~85% → ~17.6% → łącznie ~15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val:   {len(X_val)}   ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test)}  ({len(X_test)/len(X)*100:.1f}%)")

print("\nRozkład klas w zbiorach:")
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    c = Counter(y_split)
    print(f"  {split_name}: " + ", ".join(f"{lbl}={c.get(lbl,0)}" for lbl in LABEL_ORDER))

# ============================================================
# KROK 3: Trening modeli
# ============================================================
print("\n" + "=" * 60)
print("KROK 3: Trening modeli")
print("=" * 60)

# Model 1: Random Forest z class_weight='balanced' (radzi sobie z niezbalansowanymi danymi)
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Model 2: SVM z class_weight='balanced'
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
        probability=True
    ))
])

models = {
    "Random Forest": rf_pipeline,
    "SVM (RBF)":     svm_pipeline,
}

val_results = {}

for name, model in models.items():
    print(f"\n  Trening: {name}...")
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred, labels=LABEL_ORDER, average="macro", zero_division=0
    )
    val_results[name] = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
    print(f"    Val Accuracy:  {acc:.4f}")
    print(f"    Val Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")

# Wybór najlepszego modelu na podstawie F1
best_name = max(val_results, key=lambda n: val_results[n]["f1"])
best_model = models[best_name]
print(f"\n  Najlepszy model: {best_name} (F1={val_results[best_name]['f1']:.4f})")

# ============================================================
# KROK 4: Ewaluacja na zbiorze testowym
# ============================================================
print("\n" + "=" * 60)
print("KROK 4: Ewaluacja na zbiorze testowym")
print("=" * 60)

y_test_pred = best_model.predict(X_test)

print(f"\nModel: {best_name}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0))

# Metryki per klasa
p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
    y_test, y_test_pred, labels=LABEL_ORDER, zero_division=0
)
metrics_per_class = {}
for i, lbl in enumerate(LABEL_ORDER):
    metrics_per_class[lbl] = {
        "precision": round(float(p_per[i]), 4),
        "recall":    round(float(r_per[i]), 4),
        "f1":        round(float(f1_per[i]), 4),
        "support":   int(sup_per[i]),
    }

# Cross-validation na całym zbiorze (5-fold)
print("\nCross-validation (5-fold, cały zbiór):")
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="f1_macro", n_jobs=-1)
print(f"  F1 macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# KROK 5: Wykresy
# ============================================================
print("\n" + "=" * 60)
print("KROK 5: Generowanie wykresów")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f"Ewaluacja modelu: {best_name}", fontsize=14, fontweight="bold")

# -- Wykres 1: Confusion Matrix --
cm = confusion_matrix(y_test, y_test_pred, labels=LABEL_ORDER)
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax1)
ax1.set_xlabel("Predykcja")
ax1.set_ylabel("Wartość rzeczywista")
ax1.set_title("Macierz pomyłek (Confusion Matrix)")
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

# -- Wykres 2: Precision / Recall / F1 per klasa --
ax2 = axes[1]
x_pos   = np.arange(len(LABEL_ORDER))
width   = 0.25
colors  = ["#3498db", "#2ecc71", "#e74c3c"]
metrics = ["precision", "recall", "f1"]
labels_m = ["Precision", "Recall", "F1"]

for j, (metric, color, lbl_m) in enumerate(zip(metrics, colors, labels_m)):
    vals = [metrics_per_class[lbl][metric] for lbl in LABEL_ORDER]
    bars = ax2.bar(x_pos + j * width, vals, width, label=lbl_m, color=color, alpha=0.85)
    ax2.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

ax2.set_xticks(x_pos + width)
ax2.set_xticklabels(LABEL_ORDER)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Wartość")
ax2.set_title("Precision / Recall / F1 per klasa")
ax2.legend()
ax2.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Próg 0.7")

# -- Wykres 3: Porównanie modeli (Val F1) --
ax3 = axes[2]
model_names = list(val_results.keys())
f1_vals     = [val_results[n]["f1"] for n in model_names]
acc_vals    = [val_results[n]["accuracy"] for n in model_names]
x3 = np.arange(len(model_names))
w3 = 0.35
b1 = ax3.bar(x3 - w3/2, f1_vals,  w3, label="F1 macro",  color="#9b59b6", alpha=0.85)
b2 = ax3.bar(x3 + w3/2, acc_vals, w3, label="Accuracy", color="#1abc9c", alpha=0.85)
ax3.bar_label(b1, fmt="%.3f", fontsize=9, padding=3)
ax3.bar_label(b2, fmt="%.3f", fontsize=9, padding=3)
ax3.set_xticks(x3)
ax3.set_xticklabels(model_names)
ax3.set_ylim(0, 1.15)
ax3.set_ylabel("Wartość")
ax3.set_title("Porównanie modeli (zbiór walidacyjny)")
ax3.legend()

plt.tight_layout()
out_eval = OUTPUT_DIR / "model_evaluation.png"
plt.savefig(out_eval, dpi=150, bbox_inches="tight")
print(f"  Wykres ewaluacji zapisany: {out_eval}")

# -- Wykres 4: Ważność cech (Random Forest) --
if best_name == "Random Forest":
    rf_clf = best_model.named_steps["clf"]
    importances = rf_clf.feature_importances_
else:
    # Jeśli SVM wygrał, trenujemy RF tylko do ważności cech
    rf_tmp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                        random_state=42, n_jobs=-1))
    ])
    rf_tmp.fit(X_train, y_train)
    importances = rf_tmp.named_steps["clf"].feature_importances_

fig2, ax4 = plt.subplots(figsize=(10, 7))
sorted_idx = np.argsort(importances)[::-1]
sorted_features = [FEATURE_COLS[i] for i in sorted_idx]
sorted_vals     = importances[sorted_idx]

colors_imp = ["#e74c3c" if v > np.mean(importances) else "#3498db" for v in sorted_vals]
bars_imp = ax4.barh(sorted_features[::-1], sorted_vals[::-1], color=colors_imp[::-1])
ax4.set_xlabel("Ważność cechy (Feature Importance)")
ax4.set_title("Ważność cech geometrycznych (Random Forest)\nCzerwony = powyżej średniej")
ax4.axvline(x=np.mean(importances), color="gray", linestyle="--", alpha=0.7, label="Średnia")
ax4.legend()
plt.tight_layout()
out_imp = OUTPUT_DIR / "feature_importance.png"
plt.savefig(out_imp, dpi=150, bbox_inches="tight")
print(f"  Wykres ważności cech zapisany: {out_imp}")

# ============================================================
# KROK 6: Analiza błędów (False Positives / False Negatives)
# ============================================================
print("\n" + "=" * 60)
print("KROK 6: Analiza błędów")
print("=" * 60)

# Znajdź indeksy błędnych predykcji w zbiorze testowym
test_indices = df.index[
    df.index.isin(
        df.sample(frac=0.15, random_state=42).index
    )
]

errors = []
for true_lbl, pred_lbl in zip(y_test, y_test_pred):
    if true_lbl != pred_lbl:
        errors.append((true_lbl, pred_lbl))

error_counter = Counter(errors)
print(f"\nŁączna liczba błędnych predykcji: {len(errors)} / {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")
print("\nTypy błędów (prawdziwa → predykowana):")
for (true_lbl, pred_lbl), cnt in sorted(error_counter.items(), key=lambda x: -x[1]):
    print(f"  {true_lbl:15s} → {pred_lbl:15s}: {cnt}")

# False Positives / False Negatives per klasa
print("\nFalse Positives / False Negatives per klasa:")
for lbl in LABEL_ORDER:
    tp = sum(1 for t, p in zip(y_test, y_test_pred) if t == lbl and p == lbl)
    fp = sum(1 for t, p in zip(y_test, y_test_pred) if t != lbl and p == lbl)
    fn = sum(1 for t, p in zip(y_test, y_test_pred) if t == lbl and p != lbl)
    print(f"  {lbl:15s}: TP={tp}, FP={fp}, FN={fn}")

# ============================================================
# KROK 7: Zapis wyników do JSON
# ============================================================
summary = {
    "best_model": best_name,
    "dataset_size": len(df),
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "class_distribution": {k: int(v) for k, v in label_counts.items()},
    "test_accuracy": round(float(accuracy_score(y_test, y_test_pred)), 4),
    "cross_val_f1_mean": round(float(cv_scores.mean()), 4),
    "cross_val_f1_std":  round(float(cv_scores.std()),  4),
    "metrics_per_class": metrics_per_class,
    "val_comparison": {n: {k: round(v, 4) for k, v in r.items()}
                       for n, r in val_results.items()},
    "top_features": [
        {"feature": FEATURE_COLS[i], "importance": round(float(importances[i]), 4)}
        for i in sorted_idx[:5]
    ],
    "error_analysis": {
        "total_errors": len(errors),
        "error_rate": round(len(errors)/len(y_test), 4),
        "error_types": {f"{t}->{p}": cnt for (t, p), cnt in error_counter.items()}
    }
}

json_path = OUTPUT_DIR / "model_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nWyniki zapisane: {json_path}")

# Zapis modelu
model_path = OUTPUT_DIR / "best_model.pkl"
joblib.dump(best_model, model_path)
print(f"Model zapisany: {model_path}")

print("\n" + "=" * 60)
print("PODSUMOWANIE KOŃCOWE")
print("=" * 60)
print(f"  Najlepszy model:    {best_name}")
print(f"  Test Accuracy:      {accuracy_score(y_test, y_test_pred):.4f}")
p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average="macro", zero_division=0
)
print(f"  Test Precision:     {p_mac:.4f}")
print(f"  Test Recall:        {r_mac:.4f}")
print(f"  Test F1 (macro):    {f1_mac:.4f}")
print(f"  CV F1 (5-fold):     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\nWykresy zapisane w: {OUTPUT_DIR}")
print("\nGotowe! Masz teraz wszystkie wyniki do raportu Etap 2.")