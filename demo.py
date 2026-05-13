import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import mediapipe as mp
from logic import DiagnosticEngine
from pathlib import Path


class UnifiedDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("System Diagnostyczny v1.0 - Etap 3")
        self.root.geometry("1100x850")
        self.root.configure(bg="#2c3e50")

        # Inicjalizacja silnika v1.4 [cite: 12, 29]
        try:
            self.engine = DiagnosticEngine(
                "exploration_results_v1_4/final_model_v1_4.pkl",
                "face_landmarker.task"
            )
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można załadować modelu: {e}")
            self.root.destroy()
            return

        self.xml_root = r"C:\Users\danil\PycharmProjects\RIPO\Image_large_XML\Image_large_XML"
        self.is_camera_active = True
        self.timestamp = 0

        self.setup_ui()

        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def setup_ui(self):
        # Panel boczny
        ctrl_frame = tk.Frame(self.root, width=250, bg="#34495e")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(ctrl_frame, text="PANEL STEROWANIA", bg="#34495e", fg="white",
                 font=("Arial", 12, "bold")).pack(pady=20)

        # Przyciski trybów
        tk.Button(ctrl_frame, text="1. Ładuj zdjęcie z pliku", command=self.mode_load_file,
                  width=25, height=2).pack(pady=10, padx=10)

        tk.Button(ctrl_frame, text="2. Analizuj klatkę z kamery", command=self.mode_capture,
                  width=25, height=2, bg="#3498db", fg="white").pack(pady=10, padx=10)

        tk.Button(ctrl_frame, text="3. Walidacja z datasetu", command=self.mode_validate_dataset,
                  width=25, height=2).pack(pady=10, padx=10)

        tk.Button(ctrl_frame, text="Włącz kamerę / Reset", command=self.reset_to_camera,
                  width=25, height=2, bg="#27ae60", fg="white").pack(pady=30, padx=10)

        # Główny obszar (Zdjęcie/Kamera)
        self.display_frame = tk.Frame(self.root, bg="#ecf0f1")
        self.display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.lbl_main_view = tk.Label(self.display_frame, bg="#bdc3c7")
        self.lbl_main_view.pack(pady=20, padx=20, expand=True)

        self.lbl_result = tk.Label(self.display_frame, text="System gotowy. Wybierz akcję.",
                                   font=("Arial", 18, "bold"), bg="#ecf0f1", fg="#2c3e50")
        self.lbl_result.pack(pady=10)

    def update_camera(self):
        """Odświeżanie podglądu na żywo, jeśli tryb kamery jest aktywny."""
        if self.is_camera_active:
            success, frame = self.cap.read()
            if success:
                self.live_frame = frame.copy()
                self.display_image(frame)
        self.root.after(20, self.update_camera)

    def display_image(self, cv_img):
        """Pomocnicza funkcja do wyświetlania obrazu OpenCV w Tkinter."""
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_img)

        # Skalowanie z zachowaniem proporcji do 800x600
        img_pil.thumbnail((800, 600))
        img_tk = ImageTk.PhotoImage(img_pil)

        self.lbl_main_view.config(image=img_tk)
        self.lbl_main_view.image = img_tk

    def process_and_render(self, cv_img, stem=None, is_validation=False):
        """Główna funkcja przetwarzająca: detekcja -> ekstrakcja cech -> predykcja."""
        h, w, _ = cv_img.shape
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        # Detekcja landmarków v1.0 [cite: 32]
        result = self.engine.landmarker.detect_for_video(mp_img, self.timestamp)
        self.timestamp += 1

        if result.face_landmarks:
            print("Wykryto twarz, rozpoczynam ekstrakcję cech...")  # Debug
            df = self.engine.extract_features(result.face_landmarks[0], w, h)
            pred, probs = self.engine.predict(df)
            print(f"Predykcja: {pred}, Prawdopodobieństwa: {probs}")  # Debug

            # Wizualizacja na obrazie
            display_img = cv_img.copy()
            color_bgr = (0, 255, 0) if pred == "Normal" else (0, 165, 255) if pred == "SlightPalsy" else (0, 0, 255)

            # Wynik tekstowy
            msg = f"DIAGNOZA: {pred}"
            if is_validation and stem:
                truth = self.engine.get_ground_truth(self.xml_root, stem)
                status = "✓ OK" if pred == truth else "✗ BŁĄD"
                msg += f" | Prawda: {truth} | {status}"

            self.lbl_result.config(text=msg, fg="#c0392b" if "Strong" in pred else "#27ae60")

            # Rysowanie kropki na nosie dla potwierdzenia detekcji
            nose = result.face_landmarks[0][4]
            cv2.circle(display_img, (int(nose.x * w), int(nose.y * h)), 5, color_bgr, -1)

            # Wyświetlenie "zamrożonego" zdjęcia z wynikiem
            self.is_camera_active = False
            self.display_image(display_img)
        else:
            messagebox.showwarning("Błąd", "Nie wykryto twarzy na wybranym obrazie.")

    def mode_load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.png *.bmp *.jpeg")])
        if path:
            img = cv2.imread(path)
            self.process_and_render(img)

    def mode_capture(self):
        if hasattr(self, 'live_frame'):
            self.process_and_render(self.live_frame)

    def mode_validate_dataset(self):
        path = filedialog.askopenfilename(title="Wybierz zdjęcie z datasetu YFP")
        if path:
            img = cv2.imread(path)
            stem = Path(path).stem
            self.process_and_render(img, stem=stem, is_validation=True)

    def reset_to_camera(self):
        self.is_camera_active = True
        self.lbl_result.config(text="Podgląd kamery aktywny", fg="#2c3e50")


if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedDemoApp(root)
    root.mainloop()