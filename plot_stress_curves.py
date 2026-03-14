import os
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

KNOWN_STRESS_TYPES = ["clean", "gaussian", "mask", "dropout", "ood_shift", "combined"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def infer_stress_type_from_folder(folder_name):
    low = folder_name.lower()

    # Prefer exact stress matches at the end if possible
    for s in KNOWN_STRESS_TYPES:
        if f"_{s}_" in low or low.endswith(f"_{s}"):
            return s

    for s in KNOWN_STRESS_TYPES:
        if s in low:
            return s

    return "unknown"


def infer_group_info_from_path_and_json(path, data):
    """
    Folder pattern expected from your stress script:
    results/stress/<task>_<method>_<mode>_<stress>_<timestamp>/seed_0.json

    We group by the shared "default info":
      task + method + mode
    and use stress type as the line label.
    """
    folder = os.path.basename(os.path.dirname(path))
    parts = folder.split("_")

    task = str(data.get("task", parts[0] if len(parts) >= 1 else "unknown")).lower()
    method = str(data.get("method", parts[1] if len(parts) >= 2 else "unknown")).lower()

    # Try to infer mode from folder name.
    # Example: driver_dpp_oracle_clean_20260313-...
    mode = "unknown"
    for p in parts:
        pl = p.lower()
        if pl == "oracle" or pl.startswith("topk"):
            mode = pl
            break

    # Fallback to json fields if present
    if mode == "unknown":
        pure_oracle = data.get("pure_oracle", None)
        human_topk = data.get("human_topk", None)
        if pure_oracle is True:
            mode = "oracle"
        elif human_topk is not None:
            mode = f"topk{int(human_topk)}"

    stress = infer_stress_type_from_folder(folder)

    return task, method, mode, stress, folder


def extract_series(history, metric_name):
    """
    X-axis preference:
      oracle_queries if available
      otherwise num_queries

    Metrics:
      alignment
      avg_log_likelihood
    """
    xs = []
    ys = []

    for entry in history:
        x = entry.get("oracle_queries", entry.get("num_queries"))
        y = entry.get(metric_name)

        x = safe_float(x)
        y = safe_float(y)

        if x is None or y is None:
            continue

        xs.append(x)
        ys.append(y)

    if not xs:
        return np.array([]), np.array([])

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    order = np.argsort(xs)
    return xs[order], ys[order]


def pretty_group_title(task, method, mode):
    return f"task={task} | method={method} | mode={mode}"


def plot_metric(group_key, stress_to_path, metric_name, out_dir):
    task, method, mode = group_key

    plt.figure(figsize=(8, 5))

    preferred_order = ["clean", "gaussian", "mask", "dropout", "ood_shift", "combined"]
    remaining = sorted([k for k in stress_to_path.keys() if k not in preferred_order])
    ordered_stresses = [k for k in preferred_order if k in stress_to_path] + remaining

    plotted_any = False

    for stress in ordered_stresses:
        path = stress_to_path[stress]
        data = load_json(path)
        history = data.get("history", [])

        if metric_name == "alignment":
            xs, ys = extract_series(history, "alignment")
            ylabel = "Alignment"
            fname_metric = "alignment"
        else:
            xs, ys = extract_series(history, "avg_log_likelihood")
            ylabel = "Average Log-Likelihood"
            fname_metric = "loglikelihood"

        if xs.size == 0 or ys.size == 0:
            continue

        plt.plot(xs, ys, linewidth=2, label=stress)
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    title = pretty_group_title(task, method, mode)
    plt.title(title)
    plt.xlabel("Oracle Queries")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{task}_{method}_{mode}_{fname_metric}.png"
    out_path = os.path.join(out_dir, out_name)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[Saved] {out_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    root = os.path.join("results", "stress")
    pattern = os.path.join(root, "*", "seed_0.json")
    files = sorted(glob.glob(pattern))

    if not files:
        raise SystemExit(f"No files found matching: {pattern}")

    # Group by (task, method, mode), line labels by stress type
    grouped = defaultdict(dict)

    for path in files:
        data = load_json(path)
        task, method, mode, stress, folder = infer_group_info_from_path_and_json(path, data)

        # If duplicate stress exists for same group, keep the latest modified file
        if stress in grouped[(task, method, mode)]:
            old_path = grouped[(task, method, mode)][stress]
            if os.path.getmtime(path) > os.path.getmtime(old_path):
                grouped[(task, method, mode)][stress] = path
        else:
            grouped[(task, method, mode)][stress] = path

    out_dir = os.path.join(root, "plots")

    for group_key, stress_to_path in sorted(grouped.items()):
        print(f"[Group] {group_key}")
        print(f"        stresses: {sorted(stress_to_path.keys())}")

        plot_metric(group_key, stress_to_path, "alignment", out_dir)
        plot_metric(group_key, stress_to_path, "loglikelihood", out_dir)


if __name__ == "__main__":
    main()