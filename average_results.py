import argparse
import json
import os
import re
import glob
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: parsing & loading
# -----------------------------

TOPK_RE = re.compile(r"topk(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class GroupKey:
    task: str
    method: str
    tag: str  # "oracle" or "topkX"


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def infer_tag_from_filename(fname: str) -> Optional[str]:
    low = fname.lower()
    if "oracle" in low:
        return "oracle"
    m = TOPK_RE.search(low)
    if m:
        return f"topk{m.group(1)}"
    return None


def infer_group_key(data: dict, path: str) -> GroupKey:
    """
    Prefer JSON metadata. Fall back to filename parsing.

    Supported filename patterns:
      <task>_<method>_<tag>_<timestamp>.json
      <task>_<method>_<tag>.json

    Examples:
      driver_dpp_topk10_20260311-153638-717.json
      tosser_greedy_topk1_20260309-162444-458.json
    """
    fname = os.path.splitext(os.path.basename(path))[0]
    parts = fname.split("_")

    file_task = parts[0].lower() if len(parts) >= 1 else "unknown"
    file_method = parts[1].lower() if len(parts) >= 2 else "unknown"
    file_tag = parts[2].lower() if len(parts) >= 3 else "unknown"

    task = str(data.get("task") or file_task).lower()
    method = str(data.get("method") or file_method).lower()

    pure_oracle = bool(data.get("pure_oracle", False))
    human_topk = data.get("human_topk", None)

    if pure_oracle:
        tag = "oracle"
    elif human_topk is not None:
        tag = f"topk{int(human_topk)}"
    else:
        tag = infer_tag_from_filename(os.path.basename(path)) or file_tag

    return GroupKey(task=task, method=method, tag=tag)


def extract_series(history: List[dict], x_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns x, alignment, loglikelihood arrays.
    x_mode: "oracle" -> uses oracle_queries; falls back to num_queries if missing
            "all"    -> uses num_queries
    """
    xs = []
    align = []
    ll = []

    for e in history:
        if x_mode == "oracle":
            x = e.get("oracle_queries", e.get("num_queries"))
        else:
            x = e.get("num_queries")

        a = e.get("alignment", None)
        l = e.get("avg_log_likelihood", None)

        # keep points where x exists and at least one metric exists
        if x is None:
            continue

        x = safe_float(x)
        if x is None:
            continue

        a = safe_float(a) if a is not None else None
        l = safe_float(l) if l is not None else None

        xs.append(x)
        align.append(np.nan if a is None else a)
        ll.append(np.nan if l is None else l)

    if not xs:
        return np.array([]), np.array([]), np.array([])

    xs = np.asarray(xs, dtype=float)
    align = np.asarray(align, dtype=float)
    ll = np.asarray(ll, dtype=float)

    # sort by x (important for interpolation)
    order = np.argsort(xs)
    return xs[order], align[order], ll[order]


def interp_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Interpolate y onto grid. Handles NaNs by ignoring them.
    Outside range -> NaN (so mean/std ignores missing values).
    """
    if x.size == 0 or y.size == 0:
        return np.full_like(grid, np.nan, dtype=float)

    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.full_like(grid, np.nan, dtype=float)

    x2 = x[mask]
    y2 = y[mask]

    # require strictly increasing x for np.interp
    # if duplicates exist, average them
    uniq_x = []
    uniq_y = []
    for ux in np.unique(x2):
        vals = y2[x2 == ux]
        uniq_x.append(ux)
        uniq_y.append(np.mean(vals))
    x2 = np.asarray(uniq_x, dtype=float)
    y2 = np.asarray(uniq_y, dtype=float)

    if x2.size < 2:
        return np.full_like(grid, np.nan, dtype=float)

    out = np.full_like(grid, np.nan, dtype=float)
    inside = (grid >= x2.min()) & (grid <= x2.max())
    out[inside] = np.interp(grid[inside], x2, y2)
    return out


# -----------------------------
# Averaging
# -----------------------------

def average_group(file_paths: List[str], x_mode: str) -> dict:
    """
    Loads multiple runs, aligns to a common x-grid, computes mean/std.
    """
    runs = []
    for p in file_paths:
        data = load_json(p)
        hist = data.get("history", [])
        x, a, l = extract_series(hist, x_mode=x_mode)
        if x.size == 0:
            continue
        runs.append((x, a, l, p))

    if not runs:
        return {"runs": [], "history": []}

    # common grid: use integer oracle steps if oracle mode, else integer query steps
    # Use overlap range so all runs contribute:
    min_end = min(r[0].max() for r in runs)
    max_start = max(r[0].min() for r in runs)

    # average only where all runs have data (overlap region)
    grid = np.arange(int(np.ceil(max_start)), int(np.floor(min_end)) + 1, 1, dtype=float)

    A = []
    L = []
    used = []
    for x, a, l, p in runs:
        A.append(interp_to_grid(x, a, grid))
        L.append(interp_to_grid(x, l, grid))
        used.append(p)

    A = np.vstack(A)  # shape (R, T)
    L = np.vstack(L)

    mean_A = np.nanmean(A, axis=0)
    std_A = np.nanstd(A, axis=0)
    mean_L = np.nanmean(L, axis=0)
    std_L = np.nanstd(L, axis=0)

    out_hist = []
    for i in range(grid.size):
        entry = {
            "x": float(grid[i]),
            "alignment_mean": float(mean_A[i]) if not np.isnan(mean_A[i]) else None,
            "alignment_std": float(std_A[i]) if not np.isnan(std_A[i]) else None,
            "loglik_mean": float(mean_L[i]) if not np.isnan(mean_L[i]) else None,
            "loglik_std": float(std_L[i]) if not np.isnan(std_L[i]) else None,
        }
        out_hist.append(entry)

    return {
        "runs": used,
        "x_mode": x_mode,
        "history": out_hist,
    }


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Plotting (optional)
# -----------------------------

def plot_groups(avg_results: Dict[GroupKey, dict], out_dir: str, metric: str) -> None:
    """
    metric: "alignment" or "loglik"
    Plots mean curves with +/- std shading.
    """
    plt.figure()
    for key, res in sorted(avg_results.items(), key=lambda kv: kv[0].tag):
        hist = res.get("history", [])
        if not hist:
            continue

        xs = np.array([h["x"] for h in hist], dtype=float)
        if metric == "alignment":
            ys = np.array(
                [h["alignment_mean"] if h["alignment_mean"] is not None else np.nan for h in hist],
                dtype=float,
            )
            ss = np.array(
                [h["alignment_std"] if h["alignment_std"] is not None else np.nan for h in hist],
                dtype=float,
            )
            ylabel = "Alignment"
        else:
            ys = np.array(
                [h["loglik_mean"] if h["loglik_mean"] is not None else np.nan for h in hist],
                dtype=float,
            )
            ss = np.array(
                [h["loglik_std"] if h["loglik_std"] is not None else np.nan for h in hist],
                dtype=float,
            )
            ylabel = "Avg Log-Likelihood"

        label = f"{key.task} {key.method} {key.tag}"
        plt.plot(xs, ys, label=label)
        plt.fill_between(xs, ys - ss, ys + ss, alpha=0.2)

    first = next(iter(avg_results.values()))
    plt.xlabel("Oracle queries" if first.get("x_mode") == "oracle" else "Queries")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fname = os.path.join(out_dir, f"AVERAGE_{metric}.png")
    plt.savefig(fname, dpi=200)
    print(f"[Saved] {fname}")


# -----------------------------
# Input selection
# -----------------------------

def collect_json_files(args) -> List[str]:
    """
    Collect JSON files from:
      - explicit files
      - directories (recursively)
      - glob patterns (supports ** with recursive=True)

    If --inputs is omitted, falls back to walking --results_dir like before.

    Filtering:
      - only *.json
      - filename contains --pattern (substring), if provided
    """
    out: List[str] = []

    def add_file(p: str) -> None:
        if not p.lower().endswith(".json"):
            return
        fn = os.path.basename(p)
        if args.pattern and args.pattern.lower() not in fn.lower():
            return
        out.append(os.path.abspath(p))

    def add_dir(d: str) -> None:
        for root, _, files in os.walk(d):
            for fn in files:
                add_file(os.path.join(root, fn))

    if args.inputs:
        for item in args.inputs:
            item = os.path.expandvars(os.path.expanduser(item))

            if os.path.isfile(item):
                add_file(item)
            elif os.path.isdir(item):
                add_dir(item)
            else:
                matches = glob.glob(item, recursive=True)
                for m in matches:
                    if os.path.isfile(m):
                        add_file(m)
                    elif os.path.isdir(m):
                        add_dir(m)
    else:
        add_dir(args.results_dir)

    seen = set()
    deduped: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    return deduped


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results", help="Folder containing result JSON files (fallback when --inputs not set).")
    ap.add_argument("--inputs", nargs="+", default=None,
                    help=("Files/dirs/globs to average. Examples:\n"
                          "  --inputs results/context/run1.json results/context/run2.json\n"
                          "  --inputs results/context\n"
                          "  --inputs 'results/context/**/*.json'\n"
                          "If omitted, the script walks --results_dir like before."))
    ap.add_argument("--x", default="oracle", choices=["oracle", "all"], help="X-axis mode.")
    ap.add_argument("--pattern", default=None, help="Only include files whose name contains this substring. Defaults to no filename filter.")
    ap.add_argument("--out_dir", default=os.path.join("results", "averaged_context"), help="Output folder for averaged JSONs/plots.")
    ap.add_argument("--plot", action="store_true", help="Also generate mean+std plots.")
    args = ap.parse_args()

    all_files = collect_json_files(args)
    if not all_files:
        searched = args.inputs if args.inputs else [args.results_dir]
        raise SystemExit(f"No JSON files found in {searched} matching pattern '{args.pattern}'.")

    groups: Dict[GroupKey, List[str]] = defaultdict(list)
    for p in all_files:
        data = load_json(p)
        key = infer_group_key(data, p)
        groups[key].append(p)

    averaged: Dict[GroupKey, dict] = {}
    for key, paths in sorted(groups.items(), key=lambda kv: (kv[0].task, kv[0].method, kv[0].tag)):
        if len(paths) < 2:
            continue

        print(f"[Group] {key.task} | {key.method} | {key.tag}  ({len(paths)} runs)")
        res = average_group(paths, x_mode=args.x)

        out = {
            "task": key.task,
            "method": key.method,
            "tag": key.tag,
            "num_runs": len(paths),
            "runs": res["runs"],
            "x_mode": res["x_mode"],
            "history": res["history"],
        }

        out_path = os.path.join(args.out_dir, f"{key.task}_{key.method}_{key.tag}_AVG.json")
        save_json(out, out_path)
        print(f"[Saved] {out_path}")

        averaged[key] = out

    if args.plot and averaged:
        os.makedirs(args.out_dir, exist_ok=True)
        plot_groups(averaged, args.out_dir, metric="alignment")
        plot_groups(averaged, args.out_dir, metric="loglik")

    print("[Done]")


if __name__ == "__main__":
    main()