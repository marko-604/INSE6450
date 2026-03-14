import argparse
import json
import os
import time
import random

import numpy as np

from sampling import Sampler
from simulation_utils import create_env, run_algo
from metrics import ResultsTracker
from LLM_ASSISTED import get_feedback_mixed


# ------------------------------------------------------------
# Feature corruption utilities
# ------------------------------------------------------------

def corrupt_features(phi, stress_type,
                     noise_std=0.05,
                     mask_prob=0.2,
                     dropout_prob=0.1,
                     shift_scale=0.1):

    phi = np.asarray(phi, dtype=float).copy()

    if stress_type == "clean":
        return phi

    if stress_type == "gaussian":
        noise = np.random.normal(0, noise_std, size=len(phi))
        phi += noise

    elif stress_type == "mask":
        mask = np.random.rand(len(phi)) < mask_prob
        phi[mask] = 0.0

    elif stress_type == "dropout":
        if random.random() < dropout_prob:
            phi[:] = 0.0

    elif stress_type == "ood_shift":
        shift = np.random.normal(0, shift_scale, size=len(phi))
        phi += shift

    elif stress_type == "combined":

        noise = np.random.normal(0, noise_std, size=len(phi))
        phi += noise

        mask = np.random.rand(len(phi)) < mask_prob
        phi[mask] = 0.0

        if random.random() < dropout_prob:
            phi[:] = 0.0

        shift = np.random.normal(0, shift_scale, size=len(phi))
        phi += shift

    return phi


# ------------------------------------------------------------
# Compute corrupted psi
# ------------------------------------------------------------

def corrupted_psi(simulation_object,
                  input_A,
                  input_B,
                  stress_type,
                  args):

    simulation_object.feed(input_A)
    phi_A = simulation_object.get_features()

    simulation_object.feed(input_B)
    phi_B = simulation_object.get_features()

    phi_A = corrupt_features(
        phi_A,
        stress_type,
        args.noise_std,
        args.mask_prob,
        args.dropout_prob,
        args.shift_scale
    )

    phi_B = corrupt_features(
        phi_B,
        stress_type,
        args.noise_std,
        args.mask_prob,
        args.dropout_prob,
        args.shift_scale
    )

    psi = phi_A - phi_B
    return psi


# ------------------------------------------------------------
# Stress test experiment
# ------------------------------------------------------------

def run_stress_test(task,
                    method,
                    N,
                    M,
                    b,
                    args,
                    seed):

    np.random.seed(seed)
    random.seed(seed)

    simulation_object = create_env(task)

    d = simulation_object.num_of_features
    sampler = Sampler(d)

    psi_set = []
    s_set = []

    w_true = getattr(simulation_object, "w_true", None)

    tracker = ResultsTracker(
        task_name=task,
        method_name=method,
        human_topk=args.human_topk,
        pure_oracle=args.pure_oracle
    )

    num_batches = N // b

    for batch_idx in range(num_batches):

        if psi_set:
            sampler.A = np.vstack(psi_set)
            sampler.y = np.array(s_set).reshape(-1, 1)

        w_samples = sampler.sample(M)

        inputA_set, inputB_set = run_algo(
            method,
            simulation_object,
            w_samples,
            b
        )

        for j in range(b):

            psi = corrupted_psi(
                simulation_object,
                inputA_set[j],
                inputB_set[j],
                args.stress_type,
                args
            )

            if args.pure_oracle:
                psi_obs, s, source = get_feedback_mixed(
                    task=task,
                    simulation_object=simulation_object,
                    input_A=inputA_set[j],
                    input_B=inputB_set[j],
                    query_index=j,
                    batch_size=b,
                    tracker=tracker,
                    w_samples=w_samples,
                    force_human=True,
                    routing_mode="topk",
                    oracle=True
                )
            else:
                psi_obs, s, source = get_feedback_mixed(
                    task=task,
                    simulation_object=simulation_object,
                    input_A=inputA_set[j],
                    input_B=inputB_set[j],
                    query_index=j,
                    batch_size=b,
                    tracker=tracker,
                    w_samples=w_samples,
                    force_human=False,
                    routing_mode="topk",
                    oracle=False
                )

            psi_set.append(psi)
            s_set.append(s)

        mean_w = np.mean(w_samples, axis=0)

        tracker.log_iteration(
            iteration=batch_idx + 1,
            num_queries=len(psi_set),
            w_est=mean_w,
            w_true=w_true,
            psi_list=psi_set,
            s_list=s_set
        )

    return tracker


# ------------------------------------------------------------
# Result utilities
# ------------------------------------------------------------

def save_results(tracker, path):

    data = {
        "task": tracker.task,
        "method": tracker.method,
        "history": tracker.history,
        "oracle_queries": tracker.num_oracle,
        "llm_queries": tracker.num_llm
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("task")
    parser.add_argument("method")
    parser.add_argument("N", type=int)
    parser.add_argument("M", type=int)
    parser.add_argument("b", type=int)

    parser.add_argument("--stress_type",
                        default="clean",
                        choices=[
                            "clean",
                            "gaussian",
                            "mask",
                            "dropout",
                            "ood_shift",
                            "combined"
                        ])

    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--mask_prob", type=float, default=0.2)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--shift_scale", type=float, default=0.1)

    parser.add_argument("--human_topk", type=int, default=0)
    parser.add_argument("--pure_oracle", action="store_true")

    parser.add_argument("--seeds", nargs="+", type=int, default=[0])

    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    mode = "oracle" if args.pure_oracle else f"topk{args.human_topk}"

    out_dir = os.path.join(
        "results",
        "stress",
        f"{args.task}_{args.method}_{mode}_{args.stress_type}_{timestamp}"
    )

    os.makedirs(out_dir, exist_ok=True)

    all_runs = []

    for seed in args.seeds:

        print(f"Running seed {seed}")

        tracker = run_stress_test(
            args.task,
            args.method,
            args.N,
            args.M,
            args.b,
            args,
            seed
        )

        out_path = os.path.join(out_dir, f"seed_{seed}.json")

        save_results(tracker, out_path)

        all_runs.append(tracker.history)

    summary = {
        "task": args.task,
        "method": args.method,
        "stress_type": args.stress_type,
        "runs": len(args.seeds),
        "timestamp": timestamp
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Results saved to:", out_dir)


if __name__ == "__main__":
    main()