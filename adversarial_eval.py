import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation_utils import create_env
from algos import generate_psi


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_sign(x, tie_value=1):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return tie_value


def parse_w_hat(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    return np.asarray(vals, dtype=float)


def try_extract_w_hat_from_result_json(path: str) -> np.ndarray:
    """
    Tries several common key names from experiment JSON logs.
    Uses the last history entry.
    """
    with open(path, "r") as f:
        data = json.load(f)

    hist = data.get("history", [])
    if not hist:
        raise ValueError(f"No history found in {path}")

    last = hist[-1]

    candidate_keys = [
        "w_est",
        "w_hat",
        "mean_w",
        "weight_estimate",
        "estimated_weights",
    ]

    for key in candidate_keys:
        if key in last:
            arr = np.asarray(last[key], dtype=float).reshape(-1)
            if arr.size > 0:
                return arr

    raise ValueError(
        f"Could not find a weight vector in the last history entry of {path}. "
        f"Pass --w_hat manually instead."
    )


def load_eval_pairs(task: str, max_pairs: int = 500) -> np.ndarray:
    """
    Uses pre-generated control pairs from ctrl_samples/<task>.npz.
    """
    path = os.path.join("ctrl_samples", f"{task}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing evaluation pool: {path}\n"
            f"Generate it first with input_sampler.py"
        )

    data = np.load(path)
    inputs_set = data["inputs_set"]

    if max_pairs is not None and len(inputs_set) > max_pairs:
        idx = np.random.choice(len(inputs_set), size=max_pairs, replace=False)
        inputs_set = inputs_set[idx]

    return inputs_set


def oracle_labels_from_psi(psi: np.ndarray, w_true: np.ndarray) -> np.ndarray:
    """
    Labels in {-1, +1}. Ties are dropped later.
    """
    scores = psi @ w_true
    y = np.array([safe_sign(v, tie_value=0) for v in scores], dtype=int)
    return y


def model_probs(psi: np.ndarray, w_hat: np.ndarray) -> np.ndarray:
    """
    P(y = +1 | psi, w_hat)
    """
    return sigmoid(psi @ w_hat)


def model_preds_and_conf(psi: np.ndarray, w_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = model_probs(psi, w_hat)
    preds = np.where(probs >= 0.5, 1, -1)
    conf = np.maximum(probs, 1.0 - probs)
    return preds, conf


# ------------------------------------------------------------
# FGSM in feature space
# ------------------------------------------------------------

def fgsm_attack_feature_space(psi: np.ndarray, y_true: np.ndarray, w_hat: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Untargeted FGSM attack on logistic preference prediction.

    Loss:
        L = log(1 + exp(-y * (w_hat^T psi)))

    Gradient wrt psi:
        dL/dpsi = -y * sigmoid(-y * (w_hat^T psi)) * w_hat

    FGSM:
        psi_adv = psi + epsilon * sign(grad)

    Since sign(sigmoid(...)) is always positive, this effectively pushes psi
    against the oracle-consistent direction.
    """
    logits = psi @ w_hat
    factor = -y_true * sigmoid(-y_true * logits)   # shape (N,)
    grad = factor[:, None] * w_hat[None, :]        # shape (N, D)
    psi_adv = psi + epsilon * np.sign(grad)
    return psi_adv


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def robust_accuracy(psi: np.ndarray, y_true: np.ndarray, w_hat: np.ndarray) -> float:
    preds, _ = model_preds_and_conf(psi, w_hat)
    mask = (y_true != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(preds[mask] == y_true[mask]))


def avg_log_likelihood(psi: np.ndarray, y_true: np.ndarray, w_hat: np.ndarray) -> float:
    mask = (y_true != 0)
    if mask.sum() == 0:
        return float("nan")
    psi = psi[mask]
    y_true = y_true[mask]
    logits = y_true * (psi @ w_hat)
    vals = -np.log1p(np.exp(-logits))
    return float(np.mean(vals))


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """
    Returns:
        bin_centers, empirical_accuracy, avg_confidence, counts
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    emp_acc = []
    avg_conf = []
    counts = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)

        counts.append(int(mask.sum()))

        if mask.sum() == 0:
            emp_acc.append(np.nan)
            avg_conf.append(np.nan)
        else:
            emp_acc.append(float(np.mean(correct[mask])))
            avg_conf.append(float(np.mean(conf[mask])))

    return centers, np.asarray(emp_acc), np.asarray(avg_conf), np.asarray(counts)


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_robustness_curve(eps_list, acc_list, ll_list, out_dir):
    plt.figure(figsize=(7, 5))
    plt.plot(eps_list, acc_list, marker="o", linewidth=2, label="Robust Accuracy")
    plt.xlabel("Attack Strength (epsilon)")
    plt.ylabel("Accuracy")
    plt.title("Robustness Curve: Preference Accuracy vs FGSM Strength")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "robust_accuracy_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(eps_list, ll_list, marker="o", linewidth=2, label="Avg Log-Likelihood")
    plt.xlabel("Attack Strength (epsilon)")
    plt.ylabel("Average Log-Likelihood")
    plt.title("Robustness Curve: Log-Likelihood vs FGSM Strength")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "robust_loglikelihood_curve.png"), dpi=200)
    plt.close()


def plot_confidence_hist(clean_conf, adv_conf, out_dir):
    plt.figure(figsize=(7, 5))
    plt.hist(clean_conf, bins=20, alpha=0.6, label="Clean")
    plt.hist(adv_conf, bins=20, alpha=0.6, label="Adversarial")
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_histogram.png"), dpi=200)
    plt.close()


def plot_reliability_diagram(conf, correct, out_dir, name="clean", n_bins=10):
    centers, emp_acc, avg_conf, counts = reliability_bins(conf, correct, n_bins=n_bins)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect Calibration")
    mask = ~np.isnan(emp_acc)
    plt.plot(avg_conf[mask], emp_acc[mask], marker="o", linewidth=2, label=name.capitalize())

    plt.xlabel("Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title(f"Reliability Diagram ({name})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"reliability_{name}.png"), dpi=200)
    plt.close()


# ------------------------------------------------------------
# Failures
# ------------------------------------------------------------

def collect_failure_examples(psi_clean, psi_adv, y_true, w_hat, k=10):
    clean_preds, clean_conf = model_preds_and_conf(psi_clean, w_hat)
    adv_preds, adv_conf = model_preds_and_conf(psi_adv, w_hat)

    failures = []

    for i in range(len(y_true)):
        if y_true[i] == 0:
            continue

        clean_ok = (clean_preds[i] == y_true[i])
        adv_ok = (adv_preds[i] == y_true[i])

        if clean_ok and not adv_ok:
            failures.append({
                "index": int(i),
                "oracle_label": int(y_true[i]),
                "clean_pred": int(clean_preds[i]),
                "adv_pred": int(adv_preds[i]),
                "clean_conf": float(clean_conf[i]),
                "adv_conf": float(adv_conf[i]),
                "psi_clean": psi_clean[i].tolist(),
                "psi_adv": psi_adv[i].tolist(),
            })

    failures = sorted(failures, key=lambda x: -x["adv_conf"])
    return failures[:k]


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("task", type=str, help="driver or tosser")
    parser.add_argument("--max_pairs", type=int, default=500)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--result_json", type=str, default=None, help="Optional result JSON to extract final w_hat")
    parser.add_argument("--w_hat", type=str, default=None, help='Manual weights, e.g. "0.2,-0.8,0.1,0.5"')
    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()

    if args.result_json is None and args.w_hat is None:
        raise SystemExit("Provide either --result_json or --w_hat")

    if args.result_json is not None:
        w_hat = try_extract_w_hat_from_result_json(args.result_json)
    else:
        w_hat = parse_w_hat(args.w_hat)

    env = create_env(args.task)
    w_true = np.asarray(env.w_true, dtype=float).reshape(-1)

    inputs_set = load_eval_pairs(args.task, max_pairs=args.max_pairs)
    psi = generate_psi(env, inputs_set)

    y_true = oracle_labels_from_psi(psi, w_true)

    # drop ties
    mask = (y_true != 0)
    psi = psi[mask]
    y_true = y_true[mask]

    if psi.shape[1] != len(w_hat):
        raise ValueError(
            f"Dimension mismatch: psi has dim {psi.shape[1]}, but w_hat has dim {len(w_hat)}"
        )

    if args.out_dir is None:
        tag = args.task
        out_dir = os.path.join("results", "adversarial_eval", tag)
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    eps_list = []
    acc_list = []
    ll_list = []

    # clean stats
    clean_preds, clean_conf = model_preds_and_conf(psi, w_hat)
    clean_correct = (clean_preds == y_true)

    # store one adversarial set for histograms / failures
    chosen_adv_conf = None
    chosen_adv_correct = None
    chosen_adv_psi = None
    chosen_eps_for_examples = args.epsilons[-1]

    summary = {
        "task": args.task,
        "w_hat": w_hat.tolist(),
        "num_eval_pairs": int(len(y_true)),
        "results": []
    }

    for eps in args.epsilons:
        psi_adv = fgsm_attack_feature_space(psi, y_true, w_hat, eps)

        acc = robust_accuracy(psi_adv, y_true, w_hat)
        ll = avg_log_likelihood(psi_adv, y_true, w_hat)

        adv_preds, adv_conf = model_preds_and_conf(psi_adv, w_hat)
        adv_correct = (adv_preds == y_true)

        eps_list.append(float(eps))
        acc_list.append(acc)
        ll_list.append(ll)

        summary["results"].append({
            "epsilon": float(eps),
            "robust_accuracy": float(acc),
            "avg_log_likelihood": float(ll),
            "mean_confidence": float(np.mean(adv_conf)),
        })

        if np.isclose(eps, chosen_eps_for_examples):
            chosen_adv_conf = adv_conf
            chosen_adv_correct = adv_correct
            chosen_adv_psi = psi_adv

    # plots
    plot_robustness_curve(eps_list, acc_list, ll_list, out_dir)
    plot_confidence_hist(clean_conf, chosen_adv_conf, out_dir)
    plot_reliability_diagram(clean_conf, clean_correct, out_dir, name="clean")
    plot_reliability_diagram(chosen_adv_conf, chosen_adv_correct, out_dir, name=f"adv_eps_{chosen_eps_for_examples}")

    # failures
    failures = collect_failure_examples(psi, chosen_adv_psi, y_true, w_hat, k=10)

    with open(os.path.join(out_dir, "failure_examples.json"), "w") as f:
        json.dump(failures, f, indent=2)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved adversarial evaluation outputs to: {out_dir}")


if __name__ == "__main__":
    main()