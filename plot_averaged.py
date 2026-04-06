import argparse
import json
import os
import re
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

# Unique color per (method, tag)
COLOR_MAP = {
    ("greedy", "oracle"): "black",
    ("dpp", "oracle"): "red",

    ("greedy", "topk1"): "blue",
    ("dpp", "topk1"): "green",
}

# Alignment axis settings
ALIGN_Y_MIN = -1.0
ALIGN_Y_MAX = 1.0
ALIGN_Y_STEP = 0.1

ORACLE_X_MAX = 60

TOPK_RE = re.compile(r"topk(\d+)", re.IGNORECASE)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def infer_method_tag(data: dict, path: str):
    method = (data.get("method") or "").strip().lower()
    tag = (data.get("tag") or "").strip().lower()

    name = os.path.basename(path).lower()

    if not method:
        if "dpp" in name:
            method = "dpp"
        elif "greedy" in name:
            method = "greedy"
        else:
            method = "unknown"

    if not tag:
        if "oracle" in name:
            tag = "oracle"
        else:
            m = TOPK_RE.search(name)
            tag = f"topk{m.group(1)}" if m else "unknown"

    return method, tag


def extract_avg_series(data: dict, metric_prefix: str):
    hist = data.get("history", [])
    xs, means, stds = [], [], []

    mean_key = f"{metric_prefix}_mean"
    std_key = f"{metric_prefix}_std"

    for e in hist:
        x = e.get("x", None)
        m = e.get(mean_key, None)
        s = e.get(std_key, None)

        if x is None:
            continue

        x = float(x)
        m = np.nan if m is None else float(m)
        s = np.nan if s is None else float(s)

        xs.append(x)
        means.append(m)
        stds.append(s)

    if not xs:
        return np.array([]), np.array([]), np.array([])

    xs = np.asarray(xs, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    order = np.argsort(xs)
    return xs[order], means[order], stds[order]


def dedupe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_h, new_l = [], []

    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        new_h.append(h)
        new_l.append(l)

    ax.legend(new_h, new_l, fontsize=9)


def plot_mean_std(ax, x, mean, std, label, color):
    mask = ~np.isnan(mean)
    if mask.sum() == 0:
        return

    x2 = x[mask]
    m2 = mean[mask]
    s2 = std[mask]

    ax.plot(x2, m2, label=label, color=color, linewidth=1.6)

    mask_s = ~np.isnan(s2)
    if mask_s.sum() > 0:
        ax.fill_between(
            x2[mask_s],
            (m2 - s2)[mask_s],
            (m2 + s2)[mask_s],
            alpha=0.2,
            color=color,
        )


def format_axes(ax, y_label: str, is_alignment: bool):
    ax.set_xlabel("Oracle queries")
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.set_xlim(0, ORACLE_X_MAX)

    if is_alignment:
        ax.set_ylim(ALIGN_Y_MIN, ALIGN_Y_MAX)
        ax.set_yticks(np.arange(ALIGN_Y_MIN, ALIGN_Y_MAX + 1e-9, ALIGN_Y_STEP))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dir",
        default=os.path.join("results", "averaged"),
        help="Folder containing averaged *_AVG.json files.",
    )

    ap.add_argument(
        "--pattern",
        default="*_AVG.json",
        help="Glob pattern inside --dir.",
    )

    ap.add_argument(
        "--out_prefix",
        default="averaged",
        help="Prefix for output PNG filenames.",
    )

    ap.add_argument(
        "--files",
        nargs="+",
        default=None,
        metavar="FILES",
        help="Specify *_AVG.json files explicitly.",
    )

    args = ap.parse_args()

    if args.files is not None:
        paths = [os.path.abspath(os.path.expanduser(f)) for f in args.files]
    else:
        paths = sorted(glob(os.path.join(args.dir, args.pattern)))

    if not paths:
        raise SystemExit("No input files found.")

    curves = []

    for p in paths:
        data = load_json(p)

        method, tag = infer_method_tag(data, p)

        label = f"{method} {tag}"
        color = COLOR_MAP.get((method, tag), "gray")

        curves.append((label, color, data))

    # Alignment
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, color, data in curves:
        x, m, s = extract_avg_series(data, "alignment")
        plot_mean_std(ax, x, m, s, label, color)

    format_axes(ax, "Alignment", True)
    ax.set_title("Averaged Alignment vs Oracle Queries")
    dedupe_legend(ax)

    fig.tight_layout()

    out_align = f"{args.out_prefix}_alignment.png"
    fig.savefig(out_align, dpi=200)

    print(f"[Saved] {out_align}")

    # Log likelihood
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, color, data in curves:
        x, m, s = extract_avg_series(data, "loglik")
        plot_mean_std(ax, x, m, s, label, color)

    format_axes(ax, "Average log-likelihood", False)
    ax.set_title("Averaged Log-Likelihood vs Oracle Queries")
    dedupe_legend(ax)

    fig.tight_layout()

    out_ll = f"{args.out_prefix}_loglikelihood.png"
    fig.savefig(out_ll, dpi=200)

    print(f"[Saved] {out_ll}")

    plt.show()


if __name__ == "__main__":
    main()