# import torch
# import torch.nn as nn
# import numpy as np
# import pickle
# from datetime import timedelta
# import os
# import pandas as pd

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "delay_lstm_model.pth")
# SCALER_PATH = os.path.join(PROJECT_ROOT, "data", "scaler_lstm.pkl")


# DATA_PATH = "/data/lstm_final_dataset.csv"

# MODEL_PATH = "data/delay_lstm_model.pth"
# SCALER_PATH = "data/scaler_lstm.pkl"

# features_len = 6

# class LSTMModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(features_len, 64, 2,
#                             batch_first=True, dropout=0.3)
#         self.fc = nn.Linear(64, 1)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])

# model = LSTMModel()
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()

# scaler = pickle.load(open(SCALER_PATH, "rb"))

# def prediction_agent(state):
#     print("[Prediction Agent]")

 

#     features = [
#     "episode_type",
#     "time_gap_min",
#     "HR",
#     "MAP",
#     "RR",
#     "SPO2"
#     ]

#     seq_df = pd.DataFrame(state["sequence"], columns=features)
#     seq_scaled = scaler.transform(seq_df)
#     seq_tensor = torch.tensor(seq_scaled,
#                               dtype=torch.float32).unsqueeze(0)

#     with torch.no_grad():
#         pred_log = model(seq_tensor).item()

#     pred_min = np.expm1(pred_log)
#     predicted_hours = pred_min / 60

#     state["predicted_delay"] = predicted_hours
#     state["deadline"] = (
#         state["trigger_time"] +
#         timedelta(hours=predicted_hours + 0.5)
#     )

#     print("Predicted Delay:", round(predicted_hours, 2), "hr")

#     return state


    # ======================================================
# File: prediction_agent.py  [REACTIVE]
# Changes: input_size 6 → 11
#          hidden_size 64 → 128
#          fc: Linear(64,1) → Sequential(128→64→ReLU→1)
#          features list updated
# ======================================================
# ======================================================
# File: prediction_agent.py  [REACTIVE]
# Changes: is_weekend REMOVED — 10 features
#          input_size 11 → 10
# ======================================================

import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from datetime import timedelta
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

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

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = features_len,
            hidden_size = 128,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)
model.eval()

scaler = pickle.load(open(SCALER_PATH, "rb"))

def prediction_agent(state):
    print("[Prediction Agent]")

    seq_df     = pd.DataFrame(state["sequence"], columns=FEATURES)
    seq_scaled = scaler.transform(seq_df)
    seq_tensor = torch.tensor(
        seq_scaled, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        pred_log = model(seq_tensor).item()

    pred_min        = np.expm1(pred_log)
    predicted_hours = pred_min / 60

    state["predicted_delay"] = predicted_hours
    state["deadline"] = (
        state["trigger_time"] +
        timedelta(hours=predicted_hours + 0.5)
    )

    print("Predicted Delay:", round(predicted_hours, 2), "hr")

    return state