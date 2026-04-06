from simulation_utils import create_env, get_feedback, run_algo
import numpy as np
import os
import time
import tracemalloc

from sampling import Sampler
from metrics import ResultsTracker, save_results
from simulation_utils import create_env, get_feedback, run_algo, refresh_driver_w_true
from LLM_ASSISTED import (
    get_feedback_mixed,
    predictive_entropy,
    clear_baseline_history,
    reset_driver_history,
    freeze_baseline_from_current_history,
    llm_choose_driver_extra_feature,
)

from models import Driver


# ---- alignment metric (paper Eq. 15) ----
def alignment_metric(w_true, w_hat):
    """
    m = (w_true^T w_hat) / (||w_true|| * ||w_hat||)
    Returns a float or None if norms are zero.
    """
    if w_true is None or w_hat is None:
        return None

    w_true = np.asarray(w_true, dtype=float).flatten()
    w_hat = np.asarray(w_hat, dtype=float).flatten()

    if w_true.shape != w_hat.shape:
        return None

    n_true = np.linalg.norm(w_true)
    n_hat = np.linalg.norm(w_hat)
    if n_true == 0.0 or n_hat == 0.0:
        return None

    return float(np.dot(w_true, w_hat) / (n_true * n_hat))


def _compute_batch_entropies(simulation_object, inputA_set, inputB_set, w_samples):
    """
    Computes (psi_j, H_j) for each query j in the batch.
    psi_j = phi(A) - phi(B)
    H_j   = predictive entropy under w_samples
    """
    b = len(inputA_set)
    psis = []
    Hs = []

    for j in range(b):
        simulation_object.feed(inputA_set[j])
        phi_A = simulation_object.get_features()

        simulation_object.feed(inputB_set[j])
        phi_B = simulation_object.get_features()

        psi = np.asarray(phi_A, dtype=float) - np.asarray(phi_B, dtype=float)
        H = predictive_entropy(w_samples, psi)

        psis.append(psi)
        Hs.append(H)

    return psis, Hs


def _pair_bounds(simulation_object):
    """
    Build explicit bounds for a *paired* input (A,B), not a single trajectory input.
    This avoids NumPy broadcasting mistakes when feed_size=z but a query lives in 2*z.
    """
    lower_input_bound = np.array([x[0] for x in simulation_object.feed_bounds], dtype=float)
    upper_input_bound = np.array([x[1] for x in simulation_object.feed_bounds], dtype=float)

    pair_low = np.concatenate([lower_input_bound, lower_input_bound])
    pair_high = np.concatenate([upper_input_bound, upper_input_bound])
    return pair_low, pair_high


def _recompute_all_psis(simulation_object, psi_count, stored_input_pairs):
    """
    Recompute all historical psi vectors under the simulation_object's CURRENT feature set.
    Needed when the active extra feature changes across batches.

    stored_input_pairs: list of tuples (input_A, input_B)
    """
    if psi_count == 0:
        return []

    new_psi_set = []
    for idx in range(psi_count):
        input_A, input_B = stored_input_pairs[idx]

        simulation_object.feed(input_A)
        phi_A = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

        simulation_object.feed(input_B)
        phi_B = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

        psi = (phi_A - phi_B).reshape(-1)
        new_psi_set.append(psi)

    return new_psi_set

def should_trigger_update(entropies, threshold=0.55, frac_needed=0.30):
    """
    Simple drift / uncertainty trigger:
    fire if at least frac_needed of the batch has entropy >= threshold.
    """
    if entropies is None or len(entropies) == 0:
        return False

    high = sum(float(h) >= float(threshold) for h in entropies)
    return (high / len(entropies)) >= float(frac_needed)




def batch(task, method, N, M, b, human_topk=0, pure_oracle=False, warmup_oracle=0):
    """
    Batched active preference-based learning with:
      - optional oracle warmup
      - mixed or pure-oracle querying inside each batch
      - for DRIVER only: end-of-batch LLM re-check of which extra feature should be active
      - efficiency logging for milestone 4:
          * per-batch update time
          * per-query latency
          * memory usage
          * drift trigger events
          * feature-switch events

    CLI: python run.py <task> <method> N M b
    example: python run.py driver greedy 60 1000 10
    """
    warmup_n = int(max(0, min(warmup_oracle, N)))
    remaining = N - warmup_n

    if remaining % b != 0:
        print(f"(N - warmup_oracle) must be divisible by b. Got N={N}, warmup={warmup_n}, b={b}")
        return

    # ----------------------------------------------------------
    # 1) Environment & sampler
    # ----------------------------------------------------------
    simulation_object = create_env(task)

    # Reset history buffers for a clean run
    clear_baseline_history()
    reset_driver_history()

    # Determine feature dimension d
    if hasattr(simulation_object, "num_of_features"):
        d = int(simulation_object.num_of_features)
    else:
        # Fallback: infer from one feature evaluation
        simulation_object.feed(simulation_object.u0)
        phi0 = simulation_object.get_features()
        d = len(phi0)

    sampler = Sampler(d)

    # Collected data so far
    psi_set = []
    s_set = []

    # Store raw queried input pairs so we can recompute psi if driver feature-space changes
    stored_input_pairs = []

    # Ground-truth weights for alignment (if present in env)
    w_true = getattr(simulation_object, "w_true", None)

    # Tracker
    tracker = ResultsTracker(
        task_name=task,
        method_name=method,
        human_topk=human_topk,
        pure_oracle=pure_oracle,
    )

    # ----------------------------------------------------------
    # Efficiency logging containers
    # ----------------------------------------------------------
    # Note: tracemalloc tracks Python-side memory, not GPU VRAM.
    # If you want true GPU VRAM, add a torch-based logger separately.
    tracemalloc.start()

    batch_update_times = []
    batch_query_latencies = []
    batch_memory_mb = []
    batch_entropy_means = []
    drift_triggers = []
    feature_change_events = []

    # ----------------------------------------------------------
    # Warmup: PURE ORACLE queries to seed LLM context
    # ----------------------------------------------------------
    if warmup_n > 0:
        warmup_t0 = time.perf_counter()
        w_samples = sampler.sample(M)
        warmup_update_elapsed = time.perf_counter() - warmup_t0

        inputA_w, inputB_w = run_algo(
            method,
            simulation_object,
            w_samples,
            warmup_n
        )

        print(f"[warmup] collecting {warmup_n} PURE ORACLE comparisons")

        warmup_query_latencies = []
        for j in range(warmup_n):
            q_t0 = time.perf_counter()

            psi_obs, s, source = get_feedback_mixed(
                task=task,
                simulation_object=simulation_object,
                input_A=inputA_w[j],
                input_B=inputB_w[j],
                query_index=j,
                batch_size=warmup_n,
                tracker=tracker,
                w_samples=w_samples,
                force_human=True,
                routing_mode="topk",
                oracle=True,
            )

            q_elapsed = time.perf_counter() - q_t0
            warmup_query_latencies.append(q_elapsed)
            batch_query_latencies.append(q_elapsed)

            psi_set.append(np.asarray(psi_obs, dtype=float))
            s_set.append(int(s))
            stored_input_pairs.append(
                (np.asarray(inputA_w[j], dtype=float), np.asarray(inputB_w[j], dtype=float))
            )

        freeze_baseline_from_current_history()
        print(f"[warmup] baseline frozen with {warmup_n} items")

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        batch_update_times.append(float(warmup_update_elapsed))
        batch_memory_mb.append(float(peak_mem / (1024 * 1024)))
        batch_entropy_means.append(None)
        drift_triggers.append({
            "batch": 0,
            "triggered": False,
            "mean_entropy": None,
            "threshold": 0.55,
            "frac_needed": 0.30,
            "note": "warmup_oracle",
        })

        print(
            f"[warmup] update_time={warmup_update_elapsed:.4f}s | "
            f"avg_query_latency={np.mean(warmup_query_latencies):.4f}s | "
            f"python_peak_mem={peak_mem / (1024 * 1024):.2f} MB"
        )

    # number of remaining batches
    num_batches = (N - warmup_n) // b

    # ----------------------------------------------------------
    # 2) Batch loop
    # ----------------------------------------------------------
    for batch_idx in range(num_batches):
        batch_number = batch_idx + 1

        # 2.1 Update sampler with all data so far
        if psi_set:
            sampler.A = np.vstack(psi_set)
            sampler.y = np.array(s_set).reshape(-1, 1)

        # 2.2 Sample posterior weights and compute mean
        update_t0 = time.perf_counter()
        w_samples = sampler.sample(M)
        mean_w = np.mean(w_samples, axis=0)
        update_elapsed = time.perf_counter() - update_t0
        batch_update_times.append(float(update_elapsed))

        # Print normalized estimate
        if np.linalg.norm(mean_w) > 0:
            mean_w_unit = mean_w / np.linalg.norm(mean_w)
            print(f"[batch {batch_number}/{num_batches}] w-estimate = {mean_w_unit}")
        else:
            print(f"[batch {batch_number}/{num_batches}] w-estimate undefined (no data yet)")

        # 2.3 Log metrics BEFORE new queries
        tracker.log_iteration(
            iteration=batch_number,
            num_queries=len(s_set),
            w_est=mean_w,
            w_true=w_true,
            psi_list=psi_set if psi_set else None,
            s_list=s_set if s_set else None,
        )

        # 2.4 Select next batch using run_algo
        inputA_set, inputB_set = run_algo(method, simulation_object, w_samples, b)

        # 2.5 Compute entropies
        _, Hs = _compute_batch_entropies(simulation_object, inputA_set, inputB_set, w_samples)
        H_str = ", ".join([f"{h:.3f}" for h in Hs])

        drift_triggered = should_trigger_update(Hs, threshold=0.55, frac_needed=0.3)
        if drift_triggered:
            print(f"[batch {batch_idx + 1}] drift/uncertainty trigger fired")
        if not hasattr(tracker, "drift_triggers"):
            tracker.drift_triggers = []
        tracker.drift_triggers.append({
            "batch": batch_idx + 1,
            "triggered": bool(drift_triggered),
            "mean_entropy": float(np.mean(Hs)) if len(Hs) else 0.0,
        })

        mean_entropy = float(np.mean(Hs)) if len(Hs) > 0 else 0.0
        batch_entropy_means.append(mean_entropy)

        drift_triggered = should_trigger_update(Hs, threshold=0.55, frac_needed=0.30)
        drift_triggers.append({
            "batch": batch_number,
            "triggered": bool(drift_triggered),
            "mean_entropy": mean_entropy,
            "threshold": 0.55,
            "frac_needed": 0.30,
        })

        # --- Choose which indices will be ORACLE in this batch ---
        if pure_oracle:
            oracle_idx = set(range(b))
            print(f"[batch {batch_number}/{num_batches}] PURE ORACLE | Hs = [{H_str}]")
        else:
            k = int(human_topk)
            if k <= 0:
                oracle_idx = set()
            else:
                oracle_idx = set(np.argsort(Hs)[-k:].tolist())

            print(
                f"[batch {batch_number}/{num_batches}] MIXED | "
                f"oracle_topk={k}, oracle_idx={sorted(oracle_idx)} | Hs = [{H_str}] | "
                f"mean_H={mean_entropy:.3f} | drift_trigger={drift_triggered}"
            )

        # 2.6 Collect feedback for each query in batch
        this_batch_query_latencies = []

        for j in range(b):
            q_t0 = time.perf_counter()

            if j in oracle_idx:
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
                    oracle=True,
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
                    oracle=False,
                )

            q_elapsed = time.perf_counter() - q_t0
            this_batch_query_latencies.append(float(q_elapsed))
            batch_query_latencies.append(float(q_elapsed))

            psi_set.append(np.asarray(psi_obs, dtype=float))
            s_set.append(int(s))
            stored_input_pairs.append(
                (np.asarray(inputA_set[j], dtype=float), np.asarray(inputB_set[j], dtype=float))
            )

        # 2.7 END-OF-BATCH driver feature re-check
        if isinstance(simulation_object, Driver) and task == "driver":
            old_feature_id = getattr(simulation_object, "extra_feature_id", "none")

            try:
                feature_check_t0 = time.perf_counter()
                chosen_feature_id = llm_choose_driver_extra_feature(simulation_object)
                feature_check_elapsed = time.perf_counter() - feature_check_t0
            except Exception as e:
                print(f"[WARN] driver feature chooser failed ({type(e).__name__}): {e}")
                chosen_feature_id = old_feature_id
                feature_check_elapsed = None

            if chosen_feature_id != old_feature_id:
                print(
                    f"[driver][batch {batch_number}/{num_batches}] "
                    f"LLM changed extra feature: {old_feature_id} -> {chosen_feature_id}"
                )

                feature_change_events.append({
                    "batch": batch_number,
                    "old_feature": old_feature_id,
                    "new_feature": chosen_feature_id,
                    "feature_check_time_sec": float(feature_check_elapsed) if feature_check_elapsed is not None else None,
                    "num_queries_seen": len(s_set),
                })

                # activate new driver feature
                simulation_object.set_extra_feature(chosen_feature_id)
                refresh_driver_w_true(simulation_object)
                w_true = simulation_object.w_true

                # recompute all historical psis in the NEW feature space
                recomp_t0 = time.perf_counter()
                psi_set = _recompute_all_psis(
                    simulation_object=simulation_object,
                    psi_count=len(stored_input_pairs),
                    stored_input_pairs=stored_input_pairs,
                )
                recomp_elapsed = time.perf_counter() - recomp_t0

                # rebuild sampler at the NEW dimension
                d = int(simulation_object.num_of_features)
                sampler = Sampler(d)

                if psi_set:
                    sampler.A = np.vstack(psi_set)
                    sampler.y = np.array(s_set).reshape(-1, 1)

                print(
                    f"[driver][batch {batch_number}/{num_batches}] "
                    f"recomputed {len(psi_set)} psi vectors in {recomp_elapsed:.4f}s"
                )

                # Add recomputation cost into the current batch update time
                batch_update_times[-1] += float(recomp_elapsed)

            else:
                print(
                    f"[driver][batch {batch_number}/{num_batches}] "
                    f"LLM kept extra feature = {old_feature_id}"
                )

        # record memory after the batch work
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        peak_mb = float(peak_mem / (1024 * 1024))
        batch_memory_mb.append(peak_mb)

        avg_q_latency = float(np.mean(this_batch_query_latencies)) if this_batch_query_latencies else 0.0
        print(
            f"[batch {batch_number}/{num_batches}] "
            f"update_time={batch_update_times[-1]:.4f}s | "
            f"avg_query_latency={avg_q_latency:.4f}s | "
            f"python_peak_mem={peak_mb:.2f} MB"
        )

    # ----------------------------------------------------------
    # 3) Final posterior estimate after all N queries
    # ----------------------------------------------------------
    if psi_set:
        sampler.A = np.vstack(psi_set)
        sampler.y = np.array(s_set).reshape(-1, 1)

        final_update_t0 = time.perf_counter()
        w_samples = sampler.sample(M)
        mean_w = np.mean(w_samples, axis=0)
        final_update_elapsed = time.perf_counter() - final_update_t0

        if np.linalg.norm(mean_w) > 0:
            mean_w_unit = mean_w / np.linalg.norm(mean_w)
            print(f"Final w-estimate = {mean_w_unit}")

        tracker.log_iteration(
            iteration=num_batches + 1,
            num_queries=len(s_set),
            w_est=mean_w,
            w_true=w_true,
            psi_list=psi_set,
            s_list=s_set,
        )

        # keep final posterior refresh as a separate summary field
        tracker.final_update_time_sec = float(final_update_elapsed)

    # ----------------------------------------------------------
    # 4) Attach efficiency metrics to tracker
    # ----------------------------------------------------------
    tracker.batch_update_times = batch_update_times
    tracker.batch_query_latencies = batch_query_latencies
    tracker.batch_memory_mb = batch_memory_mb
    tracker.batch_entropy_means = batch_entropy_means
    tracker.drift_triggers = drift_triggers
    tracker.feature_change_events = feature_change_events

    tracker.efficiency_summary = {
        "mean_batch_update_time_sec": float(np.mean(batch_update_times)) if batch_update_times else 0.0,
        "max_batch_update_time_sec": float(np.max(batch_update_times)) if batch_update_times else 0.0,
        "mean_query_latency_sec": float(np.mean(batch_query_latencies)) if batch_query_latencies else 0.0,
        "max_query_latency_sec": float(np.max(batch_query_latencies)) if batch_query_latencies else 0.0,
        "mean_python_peak_memory_mb": float(np.mean(batch_memory_mb)) if batch_memory_mb else 0.0,
        "max_python_peak_memory_mb": float(np.max(batch_memory_mb)) if batch_memory_mb else 0.0,
        "num_drift_triggers": int(sum(1 for x in drift_triggers if x.get("triggered", False))),
        "num_feature_switches": int(len(feature_change_events)),
        "total_queries": int(len(s_set)),
        "num_batches": int(num_batches),
    }

    print("\n=== Efficiency Summary ===")
    for k, v in tracker.efficiency_summary.items():
        print(f"{k}: {v}")

    tracemalloc.stop()

    tracker.hitl_summary = {
        "total_queries": len(s_set),
        "human_or_oracle_queries": int(
            sum(1 for x in getattr(tracker, "query_sources", []) if x in ("human", "oracle"))),
        "llm_queries": int(sum(1 for x in getattr(tracker, "query_sources", []) if x == "llm")),
    }
    if tracker.hitl_summary["total_queries"] > 0:
        tracker.hitl_summary["human_fraction"] = (
                tracker.hitl_summary["human_or_oracle_queries"] / tracker.hitl_summary["total_queries"]
        )
    else:
        tracker.hitl_summary["human_fraction"] = 0.0
        # ----------------------------------------------------------
    # 5) Save metrics to JSON
    # ----------------------------------------------------------
    save_results(tracker)

def nonbatch(task, method, N, M):
    """
    Kept for completeness.
    Non-batch querying uses get_feedback (human) only.

    Also fixed to use explicit paired bounds to avoid shape mismatch bugs.
    """
    simulation_object = create_env(task)
    d = simulation_object.num_of_features

    pair_low, pair_high = _pair_bounds(simulation_object)

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []

    input_A = np.random.uniform(
        low=pair_low,
        high=pair_high,
        size=(2 * simulation_object.feed_size),
    )
    input_B = np.random.uniform(
        low=pair_low,
        high=pair_high,
        size=(2 * simulation_object.feed_size),
    )

    psi, s = get_feedback(simulation_object, input_A, input_B)
    psi_set.append(psi)
    s_set.append(s)

    for i in range(1, N):
        w_sampler.A = np.vstack(psi_set)
        w_sampler.y = np.array(s_set).reshape(-1, 1)

        w_samples = w_sampler.sample(M)

        if method == "random":
            input_A = np.random.uniform(
                low=pair_low,
                high=pair_high,
                size=(2 * simulation_object.feed_size),
            )
            input_B = np.random.uniform(
                low=pair_low,
                high=pair_high,
                size=(2 * simulation_object.feed_size),
            )
        else:
            input_A, input_B = run_algo(method, simulation_object, w_samples, 1)
            if isinstance(input_A, np.ndarray) and input_A.ndim > 1:
                input_A = input_A[0]
            if isinstance(input_B, np.ndarray) and input_B.ndim > 1:
                input_B = input_B[0]

        psi, s = get_feedback(simulation_object, input_A, input_B)
        psi_set.append(psi)
        s_set.append(s)