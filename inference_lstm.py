# ============================================================
# WHOLE CODE (Data loading -> preprocessing -> dataset build ->
#             val split -> load trained weights -> inference ->
#             save altitude vs time plot with GT & "estimated")
#
# NOTE:
# - This does NOT train.
# - It reconstructs dataset/val split deterministically, then
#   loads your saved model and runs inference on ONE val window.
#
# You ONLY need to edit:
#   - file_path
#   - weights_path
#   - out_png
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split

# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    else:
        torch.backends.cudnn.benchmark = True


# -----------------------------
# Phase NaN fill: rolling majority vote
# -----------------------------
def rolling_majority_vote(series: pd.Series, window: int) -> pd.Series:
    values = series.values
    codes, uniques = pd.factorize(values)

    codes_float = codes.astype(float)
    codes_float[codes == -1] = np.nan

    rolled = pd.Series(codes_float).rolling(window=window, center=True, min_periods=1)

    def get_window_mode(x):
        valid_votes = x[~np.isnan(x)]
        if len(valid_votes) == 0:
            return np.nan
        vals, counts = np.unique(valid_votes, return_counts=True)
        return vals[np.argmax(counts)]

    filled_codes = rolled.apply(get_window_mode, raw=True).values

    filled_series = series.copy()
    mask = ~np.isnan(filled_codes)
    filled_series.values[mask] = uniques[filled_codes[mask].astype(int)]
    return filled_series


# -----------------------------
# Dataset (same feature order as your training)
# -----------------------------
class FlightTimeSeriesDataset(Dataset):
    """
    Returns:
      x: FloatTensor (T, 7) normalized features [dt, Alt, GSpd, Trk, Lat, Lon, VRate]
      y: LongTensor (1,) phase label of last timestep
    """

    def __init__(
        self,
        instance_dict,
        window_size=20,
        stride=15,
        phase_col="Phase",
        stat_path="./stats.npz",
    ):
        self.window_size = window_size
        self.stride = stride
        self.instance_dict = instance_dict
        self.phase_col = phase_col
        self.stat_path = stat_path

        self.samples = []  # list of (Unique_ID, start_idx)

        self.raw_features = [
            "Altitude",
            "GroundSpeed",
            "Track",
            "Latitude",
            "Longitude",
            "VerticalRate",
        ]

        # Load or compute stats + label map
        if os.path.exists(stat_path):
            loaded = np.load(stat_path, allow_pickle=True)
            self.mean = loaded["mean"]
            self.std = loaded["std"]
            self.phase_map = loaded["phase_map"].item()
            self.idx_to_phase = {v: k for k, v in self.phase_map.items()}
        else:
            self._calculate_stats_and_map()

        # Build window index
        for unique_id, df in self.instance_dict.items():
            if len(df) < window_size:
                continue
            num_samples = (len(df) - window_size) // stride + 1
            for i in range(num_samples):
                self.samples.append((unique_id, i * stride))

    def _calculate_stats_and_map(self):
        all_data_list = []
        all_phases = set()

        for df in self.instance_dict.values():
            dt = df["Timestamp"].diff().dt.total_seconds().fillna(0).values.reshape(-1, 1)
            others = df[self.raw_features].interpolate().fillna(0).values
            combined = np.hstack([dt, others])
            all_data_list.append(combined)

            unique_phases = df[self.phase_col].dropna().unique()
            all_phases.update(unique_phases)

        full_data = np.vstack(all_data_list)
        self.mean = np.mean(full_data, axis=0)
        self.std = np.std(full_data, axis=0)
        self.std[self.std == 0] = 1.0

        sorted_phases = sorted(list(all_phases))
        self.phase_map = {phase: idx for idx, phase in enumerate(sorted_phases)}
        self.idx_to_phase = {idx: phase for phase, idx in self.phase_map.items()}

        np.savez(self.stat_path, mean=self.mean, std=self.std, phase_map=self.phase_map)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        unique_id, start_idx = self.samples[idx]
        df = self.instance_dict[unique_id]
        end_idx = start_idx + self.window_size
        subset = df.iloc[start_idx:end_idx]

        dt = subset["Timestamp"].diff().dt.total_seconds().fillna(0).values.reshape(-1, 1)
        others = subset[self.raw_features].interpolate().fillna(0).values
        raw_x = np.hstack([dt, others])

        x = (raw_x - self.mean) / self.std

        last_phase_str = subset[self.phase_col].iloc[-1]
        y = self.phase_map.get(last_phase_str, -1)

        return torch.FloatTensor(x), torch.LongTensor([int(y)])

    def decode_label(self, idx: int) -> str:
        return self.idx_to_phase.get(idx, "Unknown")


# -----------------------------
# Model (same as training)
# -----------------------------
class FlightPhaseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        last_out = out[:, -1, :]
        return self.fc(last_out)


def denormalize(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x_norm * std + mean


@torch.no_grad()
def infer_and_save_altitude_plot(
    model,
    dataset,
    val_subset,
    instance_dict,
    device,
    out_png="val_window_altitude_gt_vs_est.png",
    pick_idx_in_val=0,
):
    # 1) sample from val subset
    x_norm, y_gt = val_subset[pick_idx_in_val]          # (T,7), (1,)
    x_norm_b = x_norm.unsqueeze(0).to(device)           # (1,T,7)
    y_gt_int = int(y_gt.item())

    # 2) infer phase
    logits = model(x_norm_b)
    pred_int = int(torch.argmax(logits, dim=1).item())

    gt_phase = dataset.decode_label(y_gt_int)
    pred_phase = dataset.decode_label(pred_int)

    # 3) map back to window location
    base_idx = val_subset.indices[pick_idx_in_val]      # index into dataset.samples
    unique_id, start_idx = dataset.samples[base_idx]
    df = instance_dict[unique_id]
    end_idx = start_idx + dataset.window_size
    subset_df = df.iloc[start_idx:end_idx].copy()

    # 4) GT altitude + time
    time = subset_df["Timestamp"].values
    alt_gt = (
        subset_df["Altitude"]
        .interpolate()
        .fillna(method="ffill")
        .fillna(method="bfill")
        .fillna(0)
        .values
    )

    # 5) "Estimated" altitude: denormalized altitude channel from input
    x_norm_np = x_norm.detach().cpu().numpy()
    raw_x = denormalize(x_norm_np, dataset.mean, dataset.std)
    alt_est = raw_x[:, 1]  # [dt, Altitude, ...]

    # 6) plot
    plt.figure()
    plt.plot(time, alt_gt, label=f"GT Altitude (GT phase={gt_phase})")
    plt.plot(time, alt_est, label=f"Estimated Altitude (Pred phase={pred_phase})")
    plt.xlabel("Time")
    plt.ylabel("Altitude")
    plt.title(f"Unique_ID={unique_id} | window={start_idx}:{end_idx} | GT={gt_phase} Pred={pred_phase}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"✅ Saved plot: {out_png}")
    print(f"   Unique_ID: {unique_id}")
    print(f"   Window: [{start_idx}:{end_idx}]")
    print(f"   GT Phase: {gt_phase} | Pred Phase: {pred_phase}")


def main():
    seed_everything(42, deterministic=True)

    # =========================
    # EDIT THESE PATHS
    # =========================
    file_path = "/scratch2/tjgus0408/CMU/data/Labeled_OpenAP.csv"
    weights_path = "best_flight_phase_lstm_20history.pth"
    out_png = "val_window_altitude_gt_vs_est.png"

    # =========================
    # Preprocessing params (match training)
    # =========================
    selected_columns = [
        "HexIdent",
        "Date_MSG_Generated",
        "Time_MSG_Generated",
        "Date_MSG_Logged",
        "Time_MSG_Logged",
        "Altitude",
        "GroundSpeed",
        "Track",
        "Latitude",
        "Longitude",
        "VerticalRate",
        "IsOnGround",
        "Phase",
    ]

    MIN_LEN_PER_HEX = 20
    GAP_MINUTES = 30
    MIN_LEN_PER_INSTANCE = 20
    PHASE_VOTE_WINDOW = 15

    WINDOW_SIZE = 20
    STRIDE = 15
    TRAIN_RATIO = 0.8

    # Model params (must match training)
    INPUT_SIZE = 7
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # choose which val example to plot
    PICK_VAL_INDEX = 0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Load CSV
    # =========================
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path, usecols=selected_columns)
    df["Timestamp"] = pd.to_datetime(df["Date_MSG_Generated"] + " " + df["Time_MSG_Generated"])

    # =========================
    # Filter short HexIdent
    # =========================
    counts = df.groupby("HexIdent").size()
    valid_hex_ids = counts[counts > MIN_LEN_PER_HEX].index
    df_filtered = df[df["HexIdent"].isin(valid_hex_ids)].copy()

    # =========================
    # Segment by time gaps -> Unique_ID
    # =========================
    df_filtered = df_filtered.sort_values(by=["HexIdent", "Timestamp"])
    df_filtered["time_diff"] = df_filtered.groupby("HexIdent")["Timestamp"].diff()
    threshold = pd.Timedelta(minutes=GAP_MINUTES)
    df_filtered["is_new_segment"] = (df_filtered["time_diff"] > threshold).fillna(False)
    df_filtered["segment_id"] = df_filtered.groupby("HexIdent")["is_new_segment"].cumsum()
    df_filtered["Unique_ID"] = df_filtered["HexIdent"] + "_" + df_filtered["segment_id"].astype(str)

    df_final = df_filtered.drop(columns=["time_diff", "is_new_segment", "segment_id"])

    # instance_dict
    instance_dict = {k: v for k, v in df_final.groupby("Unique_ID")}

    # remove too short instances
    inst_counts = df_final.groupby("Unique_ID").size()
    valid_instances = inst_counts[inst_counts > MIN_LEN_PER_INSTANCE].index
    df_final_clean = df_final[df_final["Unique_ID"].isin(valid_instances)]
    instance_dict = {k: v for k, v in df_final_clean.groupby("Unique_ID")}

    # =========================
    # Fill Phase NaNs per instance
    # =========================
    for uid, df_inst in instance_dict.items():
        if df_inst["Phase"].isna().sum() > 0:
            voted = rolling_majority_vote(df_inst["Phase"], PHASE_VOTE_WINDOW)
            df_inst["Phase"] = df_inst["Phase"].fillna(voted)
            if df_inst["Phase"].isna().sum() > 0:
                df_inst["Phase"] = df_inst["Phase"].ffill().bfill()

    # =========================
    # Build dataset + deterministic split
    # =========================
    dataset = FlightTimeSeriesDataset(
        instance_dict=instance_dict,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        phase_col="Phase",
        stat_path="./stats.npz",
    )

    dataset_size = len(dataset)
    train_size = int(dataset_size * TRAIN_RATIO)
    val_size = dataset_size - train_size

    g = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

    # =========================
    # Build model + load weights
    # =========================
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights not found: {weights_path}")

    num_classes = len(dataset.phase_map)
    model = FlightPhaseLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    # =========================
    # Inference + save plot
    # =========================
    infer_and_save_altitude_plot(
        model=model,
        dataset=dataset,
        val_subset=val_dataset,
        instance_dict=instance_dict,
        device=DEVICE,
        out_png=out_png,
        pick_idx_in_val=PICK_VAL_INDEX,
    )


if __name__ == "__main__":
    main()
