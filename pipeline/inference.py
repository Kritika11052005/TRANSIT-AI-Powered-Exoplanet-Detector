"""
Loads the trained model + scalers once, exposes predict() for the app.
"""
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.model_def import TransitClassifier

MODEL_DIR = "models/transit_model"   # matches your repo structure


class TransitPredictor:
    def __init__(self, model_dir=MODEL_DIR, device="cpu"):
        self.device = torch.device(device)

        with open(f"{model_dir}/model_config.json") as f:
            self.config = json.load(f)

        self.model = TransitClassifier(self.config).to(self.device)
        state_dict = torch.load(f"{model_dir}/best_model.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        with open(f"{model_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{model_dir}/reg_scaler.pkl", "rb") as f:
            self.reg_scaler = pickle.load(f)

        self.label_classes = np.load(f"{model_dir}/label_classes.npy", allow_pickle=True)

    def predict(self, profile_200, scalar_8_raw):
        """
        profile_200  : np.ndarray (200,) — phase-folded, zero-mean unit-std profile
        scalar_8_raw : np.ndarray (8,)  — RAW scalars:
                       [sde, snr, period, duration_hr, n_transits,
                        true_depth_ppm, secondary_depth_ppm, has_secondary]
        """
        with torch.no_grad():
            sc_scaled = self.scaler.transform(scalar_8_raw.reshape(1, -1)).astype(np.float32)
            prof_t = torch.tensor(profile_200, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            scal_t = torch.tensor(sc_scaled, dtype=torch.float32).to(self.device)

            logits, reg_out, attn_w = self.model(prof_t, scal_t)
            probs   = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            reg_raw = self.reg_scaler.inverse_transform(reg_out.cpu().numpy()).squeeze()
            attn    = attn_w.squeeze().cpu().numpy()   # (25, 25)

        pred_idx   = int(probs.argmax())
        class_name = str(self.label_classes[pred_idx])
        confidence = float(probs[pred_idx])
        probs_dict = {str(self.label_classes[i]): round(float(probs[i]), 4)
                      for i in range(len(self.label_classes))}

        return {
            "class_name"  : class_name,
            "confidence"  : confidence,
            "probs"       : probs_dict,
            "period_days" : round(float(reg_raw[0]), 4),
            "depth_ppm"   : round(float(reg_raw[1]), 1),
            "duration_hr" : round(float(reg_raw[2]), 3),
            "attention"   : attn,
        }


_predictor = None

def get_predictor():
    """Lazy singleton — model loads once per Space session, not per request."""
    global _predictor
    if _predictor is None:
        _predictor = TransitPredictor()
    return _predictor