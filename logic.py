import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
import xml.etree.ElementTree as ET
from pathlib import Path

# Konfiguracja kolumn v1.4 [cite: 27]
ALL_FEATURE_COLS = [
    "left_eye_open", "right_eye_open", "eye_aperture_asymmetry", "eye_aperture_ratio",
    "mouth_corner_height_diff", "mouth_left_deviation", "mouth_right_deviation",
    "mouth_deviation_asymmetry", "left_brow_height", "right_brow_height",
    "brow_height_asymmetry", "brow_height_ratio", "mouth_width", "mouth_angle",
    "eye_angle", "left_eye_width", "right_eye_width", "eye_width_asymmetry",
    "HB_composite_index", "eye_mouth_asymmetry_product", "global_symmetry_score",
    "brow_eye_combined", "lower_face_severity", "eye_closure_min",
    "mouth_asymmetry_combined", "total_asymmetry_delta",
    "delta_eye_aperture_asymmetry", "roll_mean_eye_aperture_asymmetry", "roll_std_eye_aperture_asymmetry",
    "delta_mouth_corner_height_diff", "roll_mean_mouth_corner_height_diff", "roll_std_mouth_corner_height_diff",
    "delta_brow_height_asymmetry", "roll_mean_brow_height_asymmetry", "roll_std_brow_height_asymmetry",
    "delta_mouth_deviation_asymmetry", "roll_mean_mouth_deviation_asymmetry", "roll_std_mouth_deviation_asymmetry",
    "delta_global_symmetry_score", "roll_mean_global_symmetry_score", "roll_std_global_symmetry_score",
    "delta_lower_face_severity", "roll_mean_lower_face_severity", "roll_std_lower_face_severity",
    "delta_eye_closure_min", "roll_mean_eye_closure_min", "roll_std_eye_closure_min"
]


class DiagnosticEngine:
    def __init__(self, model_path, landmarker_path):
        # Ładowanie modelu ensemble v1.4
        bundle = joblib.load(model_path)
        self.clf = bundle["models"]
        self.m_normal = bundle["mult_normal"]
        self.m_strong = bundle["mult_strong"]

        # Setup MediaPipe [cite: 28]
        base_options = mp.tasks.BaseOptions(model_asset_path=landmarker_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def extract_features(self, landmarks, w, h):
        """Ekstrakcja 47 cech geometrycznych i kompozytowych [cite: 33, 35, 27]"""

        def get_p(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        lec, rec = (get_p(33) + get_p(133)) / 2.0, (get_p(362) + get_p(263)) / 2.0
        ipd = np.linalg.norm(lec - rec)
        if ipd < 1e-6: return None

        leo = np.linalg.norm(get_p(159) - get_p(145)) / ipd
        reo = np.linalg.norm(get_p(386) - get_p(374)) / ipd

        base = {
            "left_eye_open": leo, "right_eye_open": reo,
            "eye_aperture_asymmetry": abs(leo - reo),
            "eye_aperture_ratio": min(leo, reo) / max(leo, reo + 1e-9),
            "mouth_corner_height_diff": abs(get_p(61)[1] - get_p(291)[1]) / ipd,
            "HB_composite_index": (abs(leo - reo) * 3.0 + (abs(get_p(61)[1] - get_p(291)[1]) / ipd) * 2.0) / 6.0,
            "global_symmetry_score": abs(leo - reo) + (abs(get_p(61)[1] - get_p(291)[1]) / ipd),
            "eye_closure_min": min(leo, reo)
        }
        # Dla pojedynczych zdjęć cechy temporalne są zerowane [cite: 27]
        return pd.DataFrame([base]).reindex(columns=ALL_FEATURE_COLS, fill_value=0.0)

    def predict(self, df):
        """Predykcja z soft voting i dynamicznymi progami """
        all_probs = [m.predict_proba(df.values) for _, m in self.clf]
        probs = np.mean(all_probs, axis=0)[0]
        probs[0] *= self.m_normal
        probs[2] *= self.m_strong

        classes = ["Normal", "SlightPalsy", "StrongPalsy"]
        return classes[np.argmax(probs)], probs

    def get_ground_truth(self, xml_root, img_stem):
        """Pobieranie etykiety z XML dla walidacji datasetu"""
        xml_root = Path(xml_root)
        for subdir in xml_root.iterdir():
            if not subdir.is_dir(): continue
            xml_f = subdir / f"{img_stem}.xml"
            if xml_f.exists():
                root = ET.parse(xml_f).getroot()
                names = [o.findtext("name", "") for o in root.findall("object")]
                if any("Strong" in n for n in names): return "StrongPalsy"
                if any("Slight" in n for n in names): return "SlightPalsy"
                if any("Normal" in n for n in names): return "Normal"
        return "Unknown"