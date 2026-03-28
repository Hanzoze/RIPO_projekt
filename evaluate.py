import os

import pandas as pd


def test_thresholds():
    # Wczytujemy dane z kalibracji
    df = pd.read_csv("kalibracja_progow.csv")

    # PRZYKŁAD: Ustalamy progi na podstawie średnich (Mean) z kalibracji
    # Np. jeśli Mean dla Mild to 0.04, a dla Moderate to 0.08,
    # progiem rozdzielającym może być 0.06.

    print("\nSugerowane progi do sekcji 7 raportu:")
    for i in range(len(df) - 1):
        threshold = (df.iloc[i]['mean'] + df.iloc[i + 1]['mean']) / 2
        print(f"Próg między {df.iloc[i]['Category']} a {df.iloc[i + 1]['Category']}: {threshold:.4f}")


if __name__ == "__main__":
    if os.path.exists("kalibracja_progow.csv"):
        test_thresholds()
    else:
        print("Najpierw uruchom calibrate.py!")