# ======================================================
# File: lstm_train_final.py
# Project: Agentic ICU Delay Detection
# Fixes:
#   - FutureWarning int64 scaling → cast to float ✅
#   - Baseline MAE added ✅
#   - subject_id split (no patient leakage) ✅
#   - Saves training log to JSON for dashboard ✅
#   - Saves evaluation plots as PNG ✅
#   - White IEEE-style monochrome plots ✅
# ======================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ─────────────────────────────────────────────
DATA_PATH = "data/lstm_final_dataset_v3.csv"
SEQ_LEN = 5
EPOCHS = 40
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ── Load data ───────────────────────────────────────────
print("\nLoading dataset")
df = pd.read_csv(DATA_PATH, parse_dates=["trigger_time"])
print("Total rows:", len(df))

df = df.dropna(subset=["delay_hours"]).copy()
print("Rows with delay:", len(df))

upper = df["delay_hours"].quantile(0.99)
df = df[df["delay_hours"] <= upper]
print("After outlier removal:", len(df))

df = df.sort_values(["stay_id", "trigger_time"])


# ── Target ──────────────────────────────────────────────
df["delay_min"] = df["delay_hours"] * 60
df["target"] = np.log1p(df["delay_min"])


# ── Features (10) ───────────────────────────────────────
FEATURES = [
    "trigger_encoded",
    "time_gap_min",
    "HR", "MAP", "RR", "SPO2",
    "hour_of_day",
    "is_night",
    "patient_age",
    "gender_encoded",
]
print("Total features:", len(FEATURES))


# ── Clean NaNs ──────────────────────────────────────────
print("\nCleaning feature NaNs")
df = df.replace([np.inf, -np.inf], np.nan)
for col in FEATURES:
    missing = df[col].isna().sum()
    if missing > 0:
        print(f"  {col}: filling {missing} NaNs")
        df[col] = df[col].fillna(df[col].median())

df["time_gap_min"] = df["time_gap_min"].clip(0, 1440)

df[FEATURES] = df[FEATURES].astype(float)
print("Data cleaning finished")


# ── Split by subject_id ─────────────────────────────────
print("\nSplitting by subject_id (no patient leakage)")
unique_patients = df["subject_id"].unique()

train_patients, temp_patients = train_test_split(
    unique_patients, test_size=0.30, random_state=42
)
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.50, random_state=42
)

train_df = df[df["subject_id"].isin(train_patients)].copy()
val_df = df[df["subject_id"].isin(val_patients)].copy()
test_df = df[df["subject_id"].isin(test_patients)].copy()

print(f"Train rows: {len(train_df)}")
print(f"Val rows:   {len(val_df)}")
print(f"Test rows:  {len(test_df)}")


# ── Scaling ─────────────────────────────────────────────
print("\nScaling features")
scaler = StandardScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
val_df[FEATURES] = scaler.transform(val_df[FEATURES])
test_df[FEATURES] = scaler.transform(test_df[FEATURES])

pickle.dump(scaler, open("data/scaler_lstm.pkl", "wb"))
print("Scaler saved → data/scaler_lstm.pkl")


# ── Build sequences ─────────────────────────────────────
print("\nBuilding sequences")

def build_sequences(data):
    X, y = [], []
    for stay_id, group in data.groupby("stay_id"):
        group = group.sort_values("trigger_time")
        if len(group) <= SEQ_LEN:
            continue
        values = group[FEATURES].values
        targets = group["target"].values
        for i in range(len(group) - SEQ_LEN):
            X.append(values[i:i + SEQ_LEN])
            y.append(targets[i + SEQ_LEN])
    return np.array(X), np.array(y)

X_train, y_train = build_sequences(train_df)
X_val, y_val = build_sequences(val_df)
X_test, y_test = build_sequences(test_df)

print(f"Train sequences: {len(X_train)}")
print(f"Val sequences:   {len(X_val)}")
print(f"Test sequences:  {len(X_test)}")


# ── Baseline MAE ────────────────────────────────────────
mean_pred_log = y_train.mean()
baseline_pred = np.full_like(y_test, mean_pred_log)
baseline_mae = mean_absolute_error(
    np.expm1(y_test), np.expm1(baseline_pred)
)
print(f"\nBaseline MAE (mean predictor): {round(baseline_mae, 2)} min")


# ── Data loaders ────────────────────────────────────────
def make_loader(X, y, shuffle=False):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return DataLoader(
        TensorDataset(Xt, yt),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)


# ── Model ───────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
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

model = LSTMModel(len(FEATURES)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
print("Model ready — input size:", len(FEATURES))


# ── Training loop ───────────────────────────────────────
best_val = float("inf")
best_epoch = 0
patience_counter = 0
train_losses = []
val_losses = []

print("\nTraining...\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(round(train_loss, 4))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(round(val_loss, 4))

    print(f"Epoch {epoch+1:02d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), "data/delay_lstm_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print(f"\nBest val loss: {round(best_val, 4)} at epoch {best_epoch}")


# ── Save training log ───────────────────────────────────
training_log = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "best_epoch": best_epoch,
    "best_val": round(best_val, 4),
    "epochs_run": len(train_losses),
}
with open("data/training_log.json", "w") as f:
    json.dump(training_log, f, indent=2)
print("Training log saved → data/training_log.json")


# ── Test evaluation ─────────────────────────────────────
print("\nEvaluating best model on test set...")
model.load_state_dict(
    torch.load("data/delay_lstm_model.pth", map_location=DEVICE)
)
model.eval()

pred_list, true_list = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        p = model(xb)
        pred_list.extend(p.cpu().numpy())
        true_list.extend(yb.numpy())

pred = np.expm1(np.array(pred_list).flatten())
true = np.expm1(np.array(true_list).flatten())

mae = mean_absolute_error(true, pred)
rmse = np.sqrt(mean_squared_error(true, pred))
r2 = r2_score(true, pred)
improvement = round((baseline_mae - mae) / baseline_mae * 100, 1)

print("\n" + "=" * 50)
print("FINAL TEST RESULTS")
print("=" * 50)
print(f"MAE  (minutes): {round(mae, 2)}")
print(f"RMSE (minutes): {round(rmse, 2)}")
print(f"MAE  (hours):   {round(mae/60, 3)}")
print(f"R²   Score:     {round(r2, 4)}")
print("-" * 50)
print(f"Baseline MAE:   {round(baseline_mae, 2)} min")
print(f"Improvement:    {improvement}% over baseline")
print("=" * 50)


# ── Save metrics ────────────────────────────────────────
metrics = {
    "mae_min": round(mae, 2),
    "rmse_min": round(rmse, 2),
    "mae_hr": round(mae/60, 3),
    "r2": round(r2, 4),
    "baseline_mae": round(baseline_mae, 2),
    "improvement": improvement,
}
with open("data/test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved → data/test_metrics.json")


# ═════════════════════════════════════════════════════════
# PLOTS — IEEE WHITE MONOCHROME STYLE
# ═════════════════════════════════════════════════════════
os.makedirs("data/plots", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "0.75",
    "text.color": "black",
    "font.family": "serif",
    "font.size": 9,
    "legend.frameon": False,
})

LINE_BLACK = "black"
LINE_DARK = "0.35"
LINE_MID = "0.50"
FILL_LIGHT = "0.85"
FILL_MED = "0.65"
FILL_DARK = "0.45"

epochs_run = list(range(1, len(train_losses) + 1))


# ── Plot 1: Loss curve ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(
    epochs_run, train_losses,
    color=LINE_BLACK, lw=1.8, marker="o", ms=4,
    linestyle="-", label="Train loss"
)
ax.plot(
    epochs_run, val_losses,
    color=LINE_DARK, lw=1.8, marker="s", ms=4,
    linestyle="--", label="Validation loss"
)
ax.axvline(
    best_epoch, color=LINE_MID, lw=1.2,
    linestyle=":", label=f"Best epoch ({best_epoch})"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss")
ax.set_title("Training and Validation Loss")
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
ax.set_xticks(epochs_run)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("data/plots/loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved → data/plots/loss_curve.png")


# ── Plot 2: Predicted vs Actual scatter ─────────────────
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    pred, true,
    alpha=0.25, s=10,
    color=FILL_DARK, edgecolors="none"
)

max_val = max(pred.max(), true.max()) + 5
ax.plot(
    [0, max_val], [0, max_val],
    color=LINE_BLACK, lw=1.2, linestyle="--",
    label="Perfect prediction"
)

ax.set_xlabel("Predicted delay (min)")
ax.set_ylabel("Actual delay (min)")
ax.set_title("Predicted vs Actual Delay")
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.legend(fontsize=8)

ax.text(
    0.05, 0.95,
    f"MAE={round(mae,1)} min, RMSE={round(rmse,1)} min, R²={round(r2,3)}",
    transform=ax.transAxes,
    fontsize=8,
    va="top"
)

plt.tight_layout()
plt.savefig("data/plots/pred_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved → data/plots/pred_vs_actual.png")


# ── Plot 3: Delay distribution ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(
    0,
    min(np.percentile(true, 95), np.percentile(pred, 95)),
    50
)

ax.hist(
    true, bins=bins, alpha=0.65, color=FILL_LIGHT,
    label="Actual delay", edgecolor="black", linewidth=0.3
)
ax.hist(
    pred, bins=bins, alpha=0.65, color=FILL_MED,
    label="Predicted delay", edgecolor="black", linewidth=0.3
)

ax.axvline(
    true.mean(), color=LINE_BLACK, lw=1.0, linestyle=":",
    label=f"Actual mean {true.mean():.0f} min"
)
ax.axvline(
    pred.mean(), color=LINE_DARK, lw=1.0, linestyle="--",
    label=f"Predicted mean {pred.mean():.0f} min"
)

ax.set_xlabel("Delay (minutes)")
ax.set_ylabel("Count")
ax.set_title("Delay Distribution — Test Set")
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("data/plots/delay_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved → data/plots/delay_distribution.png")


# ── Plot 4: Error distribution ──────────────────────────
errors = pred - true
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(
    errors, bins=60,
    color=FILL_MED, alpha=0.9,
    edgecolor="black", linewidth=0.3
)

ax.axvline(
    0, color=LINE_BLACK, lw=1.2, linestyle="--",
    label="Zero error"
)
ax.axvline(
    errors.mean(), color=LINE_DARK, lw=1.0, linestyle=":",
    label=f"Mean error {errors.mean():.1f} min"
)

ax.set_xlabel("Prediction error (min) [pred − actual]")
ax.set_ylabel("Count")
ax.set_title("Prediction Error Distribution — Test Set")
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("data/plots/error_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved → data/plots/error_distribution.png")


# ── Plot 5: MAE per delay bucket ────────────────────────
buckets = [(0, 60), (60, 180), (180, 360), (360, 720), (720, float("inf"))]
labels = ["<1 hr", "1–3 hr", "3–6 hr", "6–12 hr", ">12 hr"]
maes = []
counts = []

for lo, hi in buckets:
    mask = (true >= lo) & (true < hi)
    if mask.sum() == 0:
        maes.append(0)
        counts.append(0)
    else:
        maes.append(mean_absolute_error(true[mask], pred[mask]))
        counts.append(mask.sum())

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(
    labels, maes,
    color=[FILL_LIGHT, "0.75", FILL_MED, "0.55", FILL_DARK],
    edgecolor="black", linewidth=0.5
)

ax.set_xlabel("True delay bucket")
ax.set_ylabel("MAE (minutes)")
ax.set_title("MAE per Delay Bucket — Test Set")
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8, axis="y")

for bar, c in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"n={c}",
        ha="center",
        va="bottom",
        fontsize=7
    )

plt.tight_layout()
plt.savefig("data/plots/mae_per_bucket.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved → data/plots/mae_per_bucket.png")


print("\n" + "=" * 50)
print("All plots saved to data/plots/")
print("=" * 50)
print("\nFiles generated:")
print("  data/delay_lstm_model.pth")
print("  data/scaler_lstm.pkl")
print("  data/training_log.json")
print("  data/test_metrics.json")
print("  data/plots/loss_curve.png")
print("  data/plots/pred_vs_actual.png")
print("  data/plots/delay_distribution.png")
print("  data/plots/error_distribution.png")
print("  data/plots/mae_per_bucket.png")