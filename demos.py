from simulation_utils import create_env, get_feedback, run_algo
import numpy as np

from sampling import Sampler
from metrics import ResultsTracker, save_results

from LLM_ASSISTED import (
    get_feedback_mixed,
    predictive_entropy,
    clear_baseline_history,
    reset_driver_history,
    freeze_baseline_from_current_history,

)



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


def batch(task, method, N, M, b, human_topk=0, pure_oracle = False, warmup_oracle=10):
    """
    Batched active preference-based learning with:
      - Batch 1 (the first batch) is PURE ORACLE and is stored as BASELINE_HISTORY
      - All subsequent batches are PURE LLM (no oracle, no human)
      - LLM prompt history uses ONLY that frozen baseline batch

    CLI: python run.py <task> <method> N M b
    example: python run.py driver greedy 60 1000 10
    """
    warmup_n = int(max(0, min(warmup_oracle, N)))
    remaining = N - warmup_n

    if remaining % b != 0:
        print(f"(N - warmup_oracle) must be divisible by b. Got N={N}, warmup={warmup_n}, b={b}")
        return

    # We ignore human_topk on purpose: after baseline we want PURE LLM.
    # Keeping the argument for backward compatibility with run.py / older calls.

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
    # Warmup: PURE ORACLE queries to seed LLM context
    # ----------------------------------------------------------
    warmup_n = int(max(0, min(warmup_oracle, N)))

    if warmup_n > 0:
        # sample prior
        w_samples = sampler.sample(M)

        # select warmup queries
        inputA_w, inputB_w = run_algo(
            method,
            simulation_object,
            w_samples,
            warmup_n
        )

        print(f"[warmup] collecting {warmup_n} PURE ORACLE comparisons")

        for j in range(warmup_n):
            psi_obs, s, source = get_feedback_mixed(
                task=task,
                simulation_object=simulation_object,
                input_A=inputA_w[j],
                input_B=inputB_w[j],
                query_index=j,
                batch_size=warmup_n,
                tracker=tracker,
                w_samples=w_samples,
                force_human=True,  # force oracle
                routing_mode="topk",
                oracle=True,
            )
            psi_set.append(psi_obs)
            s_set.append(s)

        # freeze oracle warmup into LLM baseline context
        freeze_baseline_from_current_history()
        print(f"[warmup] baseline frozen with {warmup_n} items")

    # number of remaining batches
    num_batches = (N - warmup_n) // b


    # ----------------------------------------------------------
    # 2) Batch loop
    # ----------------------------------------------------------
    for batch_idx in range(num_batches):
        # 2.1 Update sampler with all data so far
        if psi_set:
            sampler.A = np.vstack(psi_set)
            sampler.y = np.array(s_set).reshape(-1, 1)

        # 2.2 Sample posterior weights and compute mean
        w_samples = sampler.sample(M)
        mean_w = np.mean(w_samples, axis=0)

        # Print normalized estimate
        if np.linalg.norm(mean_w) > 0:
            mean_w_unit = mean_w / np.linalg.norm(mean_w)
            print(f"[batch {batch_idx+1}/{num_batches}] w-estimate = {mean_w_unit}")
        else:
            print(f"[batch {batch_idx+1}/{num_batches}] w-estimate undefined (no data yet)")

        # 2.3 Log metrics BEFORE new queries
        tracker.log_iteration(
            iteration=batch_idx + 1,
            num_queries=len(s_set),
            w_est=mean_w,
            w_true=w_true,
            psi_list=psi_set if psi_set else None,
            s_list=s_set if s_set else None,
        )

        # 2.4 Select next batch using run_algo
        inputA_set, inputB_set = run_algo(method, simulation_object, w_samples, b)

        # 2.5 Compute entropies (used to pick top-k oracle queries)
        _, Hs = _compute_batch_entropies(simulation_object, inputA_set, inputB_set, w_samples)
        H_str = ", ".join([f"{h:.3f}" for h in Hs])

        # --- Choose which indices will be ORACLE in this batch ---
        if pure_oracle:
            oracle_idx = set(range(b))
            print(f"[batch {batch_idx + 1}/{num_batches}] PURE ORACLE (paper baseline) | Hs = [{H_str}]")
        else:
            k = int(human_topk)  # 0..b
            if k <= 0:
                oracle_idx = set()
            else:
                oracle_idx = set(np.argsort(Hs)[-k:].tolist())

            print(
                f"[batch {batch_idx + 1}/{num_batches}] MIXED | "
                f"oracle_topk={k}, oracle_idx={sorted(oracle_idx)} | Hs = [{H_str}]"
            )

        # 2.6 Collect feedback for each query in batch
        for j in range(b):
            if j in oracle_idx:
                # ORACLE decision
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
                    # IMPORTANT: do NOT gate logging here anymore
                    # (LLM_ASSISTED should log oracle comparisons automatically)
                )
            else:
                # LLM decision
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

            psi_set.append(psi_obs)
            s_set.append(s)

    # ----------------------------------------------------------
    # 3) Final posterior estimate after all N queries
    # ----------------------------------------------------------
    if psi_set:
        sampler.A = np.vstack(psi_set)
        sampler.y = np.array(s_set).reshape(-1, 1)
        w_samples = sampler.sample(M)
        mean_w = np.mean(w_samples, axis=0)

        if np.linalg.norm(mean_w) > 0:
            mean_w_unit = mean_w / np.linalg.norm(mean_w)
            print(f"Final w-estimate = {mean_w_unit}")

        # final metrics point
        tracker.log_iteration(
            iteration=num_batches + 1,
            num_queries=len(s_set),
            w_est=mean_w,
            w_true=w_true,
            psi_list=psi_set,
            s_list=s_set,
        )

    # ----------------------------------------------------------
    # 4) Save metrics to JSON
    # ----------------------------------------------------------
    save_results(tracker)


def nonbatch(task, method, N, M):
    """
    Kept for completeness (unchanged logic).
    Non-batch querying uses get_feedback (human) only.
    """
    simulation_object = create_env(task)
    d = simulation_object.num_of_features

    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []

    input_A = np.random.uniform(
        low=2 * np.array(lower_input_bound),
        high=2 * np.array(upper_input_bound),
        size=(2 * simulation_object.feed_size),
    )
    input_B = np.random.uniform(
        low=2 * np.array(lower_input_bound),
        high=2 * np.array(upper_input_bound),
        size=(2 * simulation_object.feed_size),
    )

    psi, s = get_feedback(simulation_object, input_A, input_B)
    psi_set.append(psi)
    s_set.append(s)

    for i in range(1, N):
        w_sampler.A = np.vstack(psi_set)
        w_sampler.y = np.array(s_set).reshape(-1, 1)

        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples, axis=0)

        if np.linalg.norm(mean_w_samples) > 0:
            print("w-estimate = {}".format(mean_w_samples / np.linalg.norm(mean_w_samples)))
        else:
            print("w-estimate undefined (no data yet)")

        input_A, input_B = run_algo(method, simulation_object, w_samples)
        psi, s = get_feedback(simulation_object, input_A, input_B)

        psi_set.append(psi)
        s_set.append(s)

    w_sampler.A = np.vstack(psi_set)
    w_sampler.y = np.array(s_set).reshape(-1, 1)
    w_samples = w_sampler.sample(M)
    mean_w_samples = np.mean(w_samples, axis=0)

    if np.linalg.norm(mean_w_samples) > 0:
        print("Final w-estimate = {}".format(mean_w_samples / np.linalg.norm(mean_w_samples)))
    else:
        print("Final w-estimate undefined")
