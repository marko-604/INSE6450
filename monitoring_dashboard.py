import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation_utils import create_env
from algos import generate_psi


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_sign(x, tie_value=1):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return tie_value


def parse_w_hat(s: str):
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    return np.asarray(vals, dtype=float)


def load_inputs(task: str, max_pairs=None):
    path = os.path.join("ctrl_samples", f"{task}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path)
    x = data["inputs_set"]
    if max_pairs is not None and len(x) > max_pairs:
        idx = np.random.choice(len(x), size=max_pairs, replace=False)
        x = x[idx]
    return x


def js_divergence_1d(a, b, bins=20, eps=1e-12):
    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if np.isclose(lo, hi):
        return 0.0

    pa, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)

    pa = pa + eps
    pb = pb + eps
    pa = pa / pa.sum()
    pb = pb / pb.sum()

    m = 0.5 * (pa + pb)

    kl_pa_m = np.sum(pa * np.log(pa / m))
    kl_pb_m = np.sum(pb * np.log(pb / m))

    return float(0.5 * (kl_pa_m + kl_pb_m))


def psi_population_stability_index(ref, cur, bins=10, eps=1e-8):
    lo = min(np.min(ref), np.min(cur))
    hi = max(np.max(ref), np.max(cur))
    if np.isclose(lo, hi):
        return 0.0

    ref_counts, edges = np.histogram(ref, bins=bins, range=(lo, hi))
    cur_counts, _ = np.histogram(cur, bins=bins, range=(lo, hi))

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def corrupt_recent_psi(psi, mode="none", noise_std=0.08, mask_prob=0.2, shift_scale=0.15):
    out = psi.copy()

    if mode == "none":
        return out

    if mode in {"gaussian", "combined"}:
        out += np.random.normal(0, noise_std, size=out.shape)

    if mode in {"mask", "combined"}:
        mask = np.random.rand(*out.shape) < mask_prob
        out[mask] = 0.0

    if mode in {"shift", "combined"}:
        out += np.random.normal(0, shift_scale, size=(1, out.shape[1]))

    return out


def feature_schema_report(psi):
    return {
        "shape": list(psi.shape),
        "has_nan": bool(np.isnan(psi).any()),
        "has_inf": bool(np.isinf(psi).any()),
        "null_rate": float(np.mean(psi == 0.0)),
    }


def model_confidence(psi, w_hat):
    probs = sigmoid(psi @ w_hat)
    conf = np.maximum(probs, 1.0 - probs)
    return conf


def model_entropy(psi, w_hat, eps=1e-12):
    probs = sigmoid(psi @ w_hat)
    probs = np.clip(probs, eps, 1.0 - eps)
    ent = -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))
    return ent


# ------------------------------------------------------------
# plotting
# ------------------------------------------------------------

def plot_feature_drift(ref_psi, cur_psi, out_dir, task):
    d = ref_psi.shape[1]

    fig, axes = plt.subplots(d, 1, figsize=(7, 2.5 * d))
    if d == 1:
        axes = [axes]

    for j in range(d):
        ax = axes[j]
        ax.hist(ref_psi[:, j], bins=20, alpha=0.6, label="baseline")
        ax.hist(cur_psi[:, j], bins=20, alpha=0.6, label="recent")
        ax.set_title(f"{task} feature {j} drift")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_feature_drift_dashboard.png"), dpi=200)
    plt.close()


def plot_confidence_dashboard(conf, ent, out_dir):
    plt.figure(figsize=(7, 4))
    plt.hist(conf, bins=20, alpha=0.8)
    plt.xlabel("confidence")
    plt.ylabel("count")
    plt.title("Recent prediction confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_confidence_histogram_monitoring.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(ent, bins=20, alpha=0.8)
    plt.xlabel("predictive entropy")
    plt.ylabel("count")
    plt.title("Recent predictive entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_entropy_histogram_monitoring.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="driver or tosser")
    parser.add_argument("--w_hat", type=str, required=True,
                        help='manual weight vector, e.g. "0.35,0.10,0.30,-0.70"')
    parser.add_argument("--baseline_size", type=int, default=400)
    parser.add_argument("--recent_size", type=int, default=100)
    parser.add_argument("--recent_mode", type=str, default="none",
                        choices=["none", "gaussian", "mask", "shift", "combined"])
    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()

    np.random.seed(0)

    w_hat = parse_w_hat(args.w_hat)

    env = create_env(args.task)
    inputs = load_inputs(args.task, max_pairs=args.baseline_size + args.recent_size)

    baseline_inputs = inputs[:args.baseline_size]
    recent_inputs = inputs[args.baseline_size:args.baseline_size + args.recent_size]

    baseline_psi = generate_psi(env, baseline_inputs)
    recent_psi = generate_psi(env, recent_inputs)
    recent_psi = corrupt_recent_psi(recent_psi, mode=args.recent_mode)

    if baseline_psi.shape[1] != len(w_hat):
        raise ValueError(
            f"Dimension mismatch: psi dim={baseline_psi.shape[1]} but w_hat dim={len(w_hat)}"
        )

    out_dir = args.out_dir or os.path.join("results", "monitoring", args.task, args.recent_mode)
    os.makedirs(out_dir, exist_ok=True)

    baseline_schema = feature_schema_report(baseline_psi)
    recent_schema = feature_schema_report(recent_psi)

    feature_stats = []
    for j in range(baseline_psi.shape[1]):
        ref = baseline_psi[:, j]
        cur = recent_psi[:, j]

        feature_stats.append({
            "feature_index": j,
            "baseline_mean": float(np.mean(ref)),
            "baseline_std": float(np.std(ref)),
            "recent_mean": float(np.mean(cur)),
            "recent_std": float(np.std(cur)),
            "js_divergence": js_divergence_1d(ref, cur),
            "psi": psi_population_stability_index(ref, cur),
        })

    conf = model_confidence(recent_psi, w_hat)
    ent = model_entropy(recent_psi, w_hat)

    summary = {
        "task": args.task,
        "recent_mode": args.recent_mode,
        "baseline_schema": baseline_schema,
        "recent_schema": recent_schema,
        "feature_stats": feature_stats,
        "recent_confidence_mean": float(np.mean(conf)),
        "recent_confidence_std": float(np.std(conf)),
        "recent_entropy_mean": float(np.mean(ent)),
        "recent_entropy_std": float(np.std(ent)),
    }

    with open(os.path.join(out_dir, "monitoring_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_feature_drift(baseline_psi, recent_psi, out_dir, args.task)
    plot_confidence_dashboard(conf, ent, out_dir)

    print(f"Saved monitoring outputs to: {out_dir}")


if __name__ == "__main__":
    main()