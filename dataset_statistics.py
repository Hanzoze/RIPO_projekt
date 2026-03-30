import xml.etree.ElementTree as ET
from pathlib import Path
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json

# ============================================================
# KONFIGURACJA ŚCIEŻEK
# ============================================================
BASE_DIR = Path(r"C:\Users\danil\PycharmProjects\RIPO")

IMAGE_ROOTS = [
    BASE_DIR / "Image" / "Image",
    BASE_DIR / "Image2" / "Image2",
    BASE_DIR / "Image3" / "Image3",
    BASE_DIR / "Image4" / "Image4",
]

XML_ROOT = BASE_DIR / "Image_large_XML" / "Image_large_XML"
OUTPUT_DIR = BASE_DIR / "exploration_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# KROK 1: Zbieranie zdjęć (z unikalnymi kluczami ścieżek)
# ============================================================
print("=" * 60)
print("KROK 1: Skanowanie zdjęć (Rekurencyjne)")
print("=" * 60)

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}
all_images = {}
image_duplicates = []

for img_root in IMAGE_ROOTS:
    if not img_root.exists():
        print(f"  [UWAGA] Folder nie istnieje: {img_root}")
        continue

    # Używamy rglob dla pełnej rekurencji
    for f in img_root.rglob("*"):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            # Tworzymy klucz: "podfolder/nazwa" (np. "1/1")
            rel_path = f.relative_to(img_root).with_suffix('')
            key = str(rel_path).replace("\\", "/")

            if key in all_images:
                image_duplicates.append((key, f, all_images[key]))

            all_images[key] = f

print(f"  Znaleziono unikalnych zdjęć: {len(all_images)}")
if image_duplicates:
    print(f"  [ALARM] Znaleziono {len(image_duplicates)} konfliktów ścieżek!")

# ============================================================
# KROK 2: Zbieranie i parsowanie XMLi
# ============================================================
print("\n" + "=" * 60)
print("KROK 2: Skanowanie i parsowanie XMLi")
print("=" * 60)

all_xmls = {}
if XML_ROOT.exists():
    for f in XML_ROOT.rglob("*.xml"):
        rel_path = f.relative_to(XML_ROOT).with_suffix('')
        key = str(rel_path).replace("\\", "/")
        all_xmls[key] = f
else:
    print(f"  [BŁĄD] Folder XML nie istnieje: {XML_ROOT}")

parsed_annotations = {}
broken_xmls = []
missing_image_for_xml = []

for key, xml_path in all_xmls.items():
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="UNKNOWN")
            pose = obj.findtext("pose", default="Unknown")
            bbox_el = obj.find("bndbox")
            bbox = None
            if bbox_el is not None:
                try:
                    bbox = {
                        "xmin": int(bbox_el.findtext("xmin")),
                        "ymin": int(bbox_el.findtext("ymin")),
                        "xmax": int(bbox_el.findtext("xmax")),
                        "ymax": int(bbox_el.findtext("ymax")),
                    }
                except (TypeError, ValueError):
                    pass
            objects.append({"name": name, "pose": pose, "bbox": bbox})

        parsed_annotations[key] = objects
        if key not in all_images:
            missing_image_for_xml.append(key)

    except ET.ParseError as e:
        broken_xmls.append((key, str(e)))

missing_xml_for_image = [k for k in all_images if k not in all_xmls]

print(f"  Poprawnie sparsowanych XMLi: {len(parsed_annotations)}")
print(f"  Zdjęcia BEZ XMLa: {len(missing_xml_for_image)}")
print(f"  XMLe BEZ zdjęcia: {len(missing_image_for_xml)}")

# ============================================================
# KROK 3: Analiza klas
# ============================================================
class_counter = collections.Counter()
pose_counter = collections.Counter()
global_labels = {}


def get_priority_label(objects):
    names = [o["name"] for o in objects]
    if any("Strong" in n for n in names): return "StrongPalsy"
    if any("Slight" in n for n in names): return "SlightPalsy"
    return "Normal" if any("Normal" in n for n in names) else "Unknown"


for key, objects in parsed_annotations.items():
    if key in all_images:
        global_labels[key] = get_priority_label(objects)
        for obj in objects:
            class_counter[obj["name"]] += 1
            pose_counter[obj["pose"]] += 1

global_counter = collections.Counter(global_labels.values())
print(f"\n  Globalna etykieta per zdjęcie:")
for lbl, cnt in sorted(global_counter.items(), key=lambda x: -x[1]):
    print(f"    {lbl:15s}: {cnt:4d}  ({cnt/len(global_labels)*100:.1f}%)")

# ============================================================
# KROK 4: Wykresy
# ============================================================
print("\n" + "=" * 60)
print("KROK 4: Generowanie wykresów")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Analiza datasetu YFP – Rozkład danych", fontsize=14, fontweight="bold")

# -- Wykres 1: klasy obiektów (bounding box) --
ax1 = axes[0]
classes = list(class_counter.keys())
counts  = list(class_counter.values())
colors  = []
for c in classes:
    if "Strong" in c:  colors.append("#e74c3c")
    elif "Slight" in c: colors.append("#e67e22")
    else:               colors.append("#2ecc71")

bars = ax1.barh(classes, counts, color=colors)
ax1.set_xlabel("Liczba obiektów")
ax1.set_title("Rozkład klas (bounding box)")
ax1.bar_label(bars, padding=3)
ax1.invert_yaxis()

# Legenda
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="StrongPalsy"),
    Patch(facecolor="#e67e22", label="SlightPalsy"),
    Patch(facecolor="#2ecc71", label="Normal"),
]
ax1.legend(handles=legend_elements, loc="lower right")

# -- Wykres 2: globalna etykieta per zdjęcie --
ax2 = axes[1]
glbl_labels = list(global_counter.keys())
glbl_counts = list(global_counter.values())
pie_colors = []
for l in glbl_labels:
    if "Strong" in l:  pie_colors.append("#e74c3c")
    elif "Slight" in l: pie_colors.append("#e67e22")
    elif "Normal" in l: pie_colors.append("#2ecc71")
    else:               pie_colors.append("#95a5a6")

wedges, texts, autotexts = ax2.pie(
    glbl_counts,
    labels=glbl_labels,
    colors=pie_colors,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 10}
)
ax2.set_title(f"Globalna etykieta per zdjęcie\n(N={len(global_labels)})")

# -- Wykres 3: dostępność danych --
ax3 = axes[2]
categories = ["Zdjęcia\nłącznie", "Zdjęcia\nz XML", "Uszkodzone\nXMLe", "Brak\nXML"]
values = [
    len(all_images),
    len(global_labels),
    len(broken_xmls),
    len(missing_xml_for_image)
]
bar_colors = ["#3498db", "#2ecc71", "#e74c3c", "#e67e22"]
bars3 = ax3.bar(categories, values, color=bar_colors, edgecolor="white")
ax3.set_title("Dostępność i jakość danych")
ax3.set_ylabel("Liczba")
ax3.bar_label(bars3, padding=3)

plt.tight_layout()
out_plot = OUTPUT_DIR / "dataset_analysis.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Wykres zapisany: {out_plot}")

# ============================================================
# KROK 5: Przykłady wizualne (po 2 z każdej klasy globalnej)
# ============================================================
print("\n" + "=" * 60)
print("KROK 5: Przykłady wizualne z bounding boxami")
print("=" * 60)

target_classes = ["Normal", "SlightPalsy", "StrongPalsy"]
BBOX_COLORS = {
    "Normal":      "#2ecc71",
    "SlightPalsy": "#e67e22",
    "StrongPalsy": "#e74c3c",
}

fig2, axes2 = plt.subplots(3, 2, figsize=(12, 14))
fig2.suptitle("Przykłady wizualne datasetu z adnotacjami", fontsize=13, fontweight="bold")

for row_idx, target_cls in enumerate(target_classes):
    samples = [stem for stem, lbl in global_labels.items() if lbl == target_cls][:2]
    for col_idx, stem in enumerate(samples):
        ax = axes2[row_idx][col_idx]
        img_path = all_images[stem]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"{target_cls}\n{stem}", fontsize=9)
            ax.axis("off")

            # Rysowanie bounding boxów
            for obj in parsed_annotations.get(stem, []):
                bbox = obj.get("bbox")
                if bbox:
                    name = obj["name"]
                    color = "#e74c3c" if "Strong" in name else \
                            "#e67e22" if "Slight" in name else "#2ecc71"
                    rect = patches.Rectangle(
                        (bbox["xmin"], bbox["ymin"]),
                        bbox["xmax"] - bbox["xmin"],
                        bbox["ymax"] - bbox["ymin"],
                        linewidth=2, edgecolor=color, facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(bbox["xmin"], bbox["ymin"] - 5,
                            f'{name}\n({obj["pose"]})',
                            color=color, fontsize=7, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="black", alpha=0.5))
        except Exception as e:
            ax.set_title(f"Błąd wczytania:\n{stem}", fontsize=8)
            ax.axis("off")
            print(f"    [UWAGA] Nie udało się wczytać {stem}: {e}")

    # Jeśli mniej niż 2 próbki
    if len(samples) < 2:
        for col_idx in range(len(samples), 2):
            axes2[row_idx][col_idx].axis("off")
            axes2[row_idx][col_idx].set_title(f"Brak próbek dla {target_cls}", fontsize=9)

plt.tight_layout()
out_examples = OUTPUT_DIR / "visual_examples.png"
plt.savefig(out_examples, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Przykłady wizualne zapisane: {out_examples}")

# ============================================================
# KROK 6: Zapis statystyk do JSON (do raportu)
# ============================================================
stats = {
    "total_images": len(all_images),
    "total_xmls": len(all_xmls),
    "parsed_xmls": len(parsed_annotations),
    "broken_xmls": len(broken_xmls),
    "missing_xml_for_image": len(missing_xml_for_image),
    "missing_image_for_xml": len(missing_image_for_xml),
    "usable_samples": len(global_labels),
    "class_distribution_objects": dict(class_counter),
    "global_label_distribution": dict(global_counter),
}
stats_path = OUTPUT_DIR / "dataset_stats.json"
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("PODSUMOWANIE")
print("=" * 60)
print(f"  Wszystkich zdjęć:           {len(all_images)}")
print(f"  Użytecznych par (img+xml):  {len(global_labels)}")
print(f"  Uszkodzone XMLe:            {len(broken_xmls)}")
print(f"  Wyniki zapisane w:          {OUTPUT_DIR}")
print("\nGotowe! Uruchom teraz step2_extract_features.py")

if image_duplicates:
    with open(OUTPUT_DIR / "path_conflicts.txt", "w") as f:
        for key, p1, p2 in image_duplicates:
            f.write(f"Klucz: {key}\n  Plik 1: {p1}\n  Plik 2: {p2}\n\n")

# Statystyki końcowe
stats = {
    "total_images": len(all_images),
    "usable_pairs": len(global_labels),
    "conflicts": len(image_duplicates)
}

with open(OUTPUT_DIR / "dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nGotowe! Użyteczne pary: {len(global_labels)}")
print(f"Sprawdź folder: {OUTPUT_DIR}")