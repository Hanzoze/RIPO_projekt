from pathlib import Path

XML_ROOT = Path(r"C:\Users\danil\PycharmProjects\RIPO\Image_large_XML\Image_large_XML")

fixed = 0
already_ok = 0
other_error = 0

print("Rozpoczynanie skanowania...")

for f in XML_ROOT.rglob("*.xml"):
    try:
        content = f.read_text(encoding="utf-8", errors="replace").strip()

        if not content.endswith("</annotation>"):
            fixed_content = content + "\n</annotation>"
            f.write_text(fixed_content, encoding="utf-8")
            fixed += 1
        else:
            already_ok += 1

    except Exception as e:
        print(f"Błąd w pliku {f.relative_to(XML_ROOT)}: {e}")
        other_error += 1

print("-" * 30)
print(f"Naprawione (dodano tag):  {fixed}")
print(f"Już były poprawne:       {already_ok}")
print(f"Błędy odczytu/zapisu:    {other_error}")