import csv
import os
import numpy as np
import yaml
from datetime import datetime

EXPERIMENTS_CSV = "experiments.csv"
CSV_HEADER = [
    "run_id", "timestamp", "dataset", "method", "indicator",
    "motion_correct", "temporal_bin", "fps", "dff_window_sec",
    "cp_model", "cp_diameter", "n_neurons", "mean_snr", "std_snr",
    "global_iou", "n_matched", "notes",
]

def get_next_run_id():
    if not os.path.exists(EXPERIMENTS_CSV) or os.path.getsize(EXPERIMENTS_CSV) == 0:
        return 1
    with open(EXPERIMENTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        ids = []
        for row in reader:
            try:
                ids.append(int(row["run_id"]))
            except (ValueError, KeyError):
                continue
    return max(ids, default=0) + 1

def log_experiment(run_id, config, method, n_neurons, snr_values,
                   global_iou_val=None, n_matched=None):
    # Ensure header exists
    if not os.path.exists(EXPERIMENTS_CSV) or os.path.getsize(EXPERIMENTS_CSV) == 0:
        with open(EXPERIMENTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    mean_snr = float(np.nanmean(snr_values)) if len(snr_values) > 0 else 0.0
    std_snr = float(np.nanstd(snr_values)) if len(snr_values) > 0 else 0.0

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": config["dataset"],
        "method": method,
        "indicator": config["indicator"],
        "motion_correct": config["motion_correct"],
        "temporal_bin": config.get("temporal_bin", ""),
        "fps": config.get("fps", ""),
        "dff_window_sec": config.get("dff_window_sec", 30.0),
        "cp_model": config.get("cp_model", "") if method == "cellpose" else "",
        "cp_diameter": config.get("cp_diameter", "") if method == "cellpose" else "",
        "n_neurons": n_neurons,
        "mean_snr": f"{mean_snr:.2f}",
        "std_snr": f"{std_snr:.2f}",
        "global_iou": f"{global_iou_val:.3f}" if global_iou_val is not None else "",
        "n_matched": n_matched if n_matched is not None else "",
        "notes": config.get("notes", ""),
    }
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerow(row)

def save_config(run_id, config, output_dir="results"):
    exp_dir = os.path.join("experiments", f"exp_{run_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Config saved: {config_path}")
