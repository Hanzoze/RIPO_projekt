import xml.etree.ElementTree as ET
from pathlib import Path

XML_ROOT = Path(r"C:\Users\danil\PycharmProjects\RIPO\Image_large_XML\Image_large_XML")

broken_file = None
print("Szukanie uszkodzonego pliku...")

for f in XML_ROOT.rglob("*.xml"):
    try:
        ET.parse(f)
    except ET.ParseError:
        broken_file = f
        break

if broken_file:
    print(f"\nAnaliza pliku: {broken_file}")
    print("-" * 50)
    lines = broken_file.read_text(encoding="utf-8", errors="replace").splitlines()

    start_line = max(0, len(lines) - 15)
    for i, line in enumerate(lines[start_line:], start_line + 1):
        print(f"{i:3}: {line}")
    print("-" * 50)
    print(f"Łączna liczba linii: {len(lines)}")
else:
    print("Nie znaleziono uszkodzonych plików (wszystkie są poprawne dla ET).")