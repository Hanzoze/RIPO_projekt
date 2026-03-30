import xml.etree.ElementTree as ET
from pathlib import Path

XML_ROOT = Path(r"C:\Users\danil\PycharmProjects\RIPO\Image_large_XML\Image_large_XML")

errors_log = []

print("Rozpoczynanie szczegółowej diagnostyki...")

for f in XML_ROOT.rglob("*.xml"):
    try:
        ET.parse(f)
    except ET.ParseError as e:
        errors_log.append(f"{f.name}: {e}")

print("\n--- PRZYKŁADY BŁĘDÓW ---")
for err in errors_log[:20]:
    print(err)

print(f"\nŁącznie błędnych plików: {len(errors_log)}")