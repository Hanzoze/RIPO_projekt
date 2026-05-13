from pathlib import Path
from tqdm import tqdm

XML_ROOT = Path(r"C:\Users\danil\PycharmProjects\RIPO\Image_large_XML\Image_large_XML")

fixed_count = 0
already_ok = 0

print("Wykonuję wymuszoną naprawę (Force Fix)...")

xml_files = list(XML_ROOT.rglob("*.xml"))

for f in tqdm(xml_files, desc="Naprawianie"):
    try:
        text = f.read_text(encoding="utf-8", errors="replace")

        if "</annoatation>" in text:
            new_text = text.replace("</annoatation>", "")

            lines = [line for line in new_text.splitlines() if line.strip()]
            final_text = "\n".join(lines)

            f.write_text(final_text, encoding="utf-8")
            fixed_count += 1
        else:
            already_ok += 1
    except Exception as e:
        print(f"Błąd w pliku {f.name}: {e}")

print("-" * 30)
print(f"Naprawione (Force Fix): {fixed_count}")
print(f"Prawidłowe/Pominięte:  {already_ok}")

print("\nGotowe! Teraz Twoje skrypty statystyk i MediaPipe powinny działać.")