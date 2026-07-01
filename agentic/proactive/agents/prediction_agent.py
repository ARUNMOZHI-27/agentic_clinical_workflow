# import torch
# import os
# import joblib
# import numpy as np

# # ==============================
# # PATH SETUP
# # ==============================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

# MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "delay_lstm_model.pth")
# SCALER_PATH = os.path.join(PROJECT_ROOT, "data", "scaler_lstm.pkl")

# # ==============================
# # LOAD MODEL + SCALER
# # ==============================

# scaler = joblib.load(SCALER_PATH)

# class LSTMModel(torch.nn.Module):
  
#     def __init__(self, input_size=6, hidden_size=64, num_layers=2):
#         super().__init__()

#         self.lstm = torch.nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True
#         )

#         self.fc = torch.nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out
# model = LSTMModel()
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()

# # ==============================
# # PREDICTION AGENT
# # ==============================

# def prediction_agent(state):
#     print("[Prediction Agent]")

#     if state.get("stop", False):
#         return state

#     seq_df = state["sequence_df"]

#     # Scale
#     seq_scaled = scaler.transform(seq_df)

#     # Convert to tensor
#     seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

#     with torch.no_grad():
#         prediction = model(seq_tensor).item()

#     state["predicted_delay_hours"] = prediction

#     print(f"Predicted Expected Delay: {round(prediction,2)} hours")

#     return state



# ======================================================
# File: prediction_agent.py  [PROACTIVE]
# Changes: input_size 6 → 11
#          hidden_size 64 → 128
#          fc updated to Sequential
#          inverse transform bug fixed (expm1 + /60)
# ======================================================
# ======================================================
# File: prediction_agent.py  [PROACTIVE]
# Changes: is_weekend REMOVED — 10 features
#          input_size 11 → 10
# ======================================================

import torch
import numpy as np
import joblib
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

MODEL_PATH  = os.path.join(PROJECT_ROOT, "data", "delay_lstm_model.pth")
SCALER_PATH = os.path.join(PROJECT_ROOT, "data", "scaler_lstm.pkl")

FEATURES = [
    "trigger_encoded",
    "time_gap_min",
    "HR", "MAP", "RR", "SPO2",
    "hour_of_day",
    "is_night",
    # is_weekend REMOVED
    "patient_age",
    "gender_encoded",
]

features_len = len(FEATURES)   # 10

class LSTMModel(torch.nn.Module):
    def __init__(self,
                 input_size  = 10,    # updated from 11
                 hidden_size = 128):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = 2,
            batch_first = True
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

scaler = joblib.load(SCALER_PATH)

model = LSTMModel()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)
model.eval()

def prediction_agent(state):
    print("[Prediction Agent]")

    if state.get("stop", False):
        return state

    seq_df     = state["sequence_df"]
    seq_scaled = scaler.transform(seq_df)
    seq_tensor = torch.tensor(
        seq_scaled, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        pred_log = model(seq_tensor).item()

    pred_min        = np.expm1(pred_log)
    predicted_hours = pred_min / 60

    state["predicted_delay_hours"] = predicted_hours

    print(f"Predicted Delay: {round(predicted_hours, 2)} hrs")

    return state