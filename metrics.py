# metrics.py
import os
import json
import time
import numpy as np


def compute_alignment(w_true, w_est):
    """
    Alignment metric m = (w_true · w_est) / (||w_true|| * ||w_est||).
    Returns None if any vector is invalid.
    """
    if w_true is None or w_est is None:
        return None

    w_true = np.asarray(w_true, dtype=float).flatten()
    w_est = np.asarray(w_est, dtype=float).flatten()

    if w_true.shape != w_est.shape:
        return None

    norm_true = np.linalg.norm(w_true)
    norm_est = np.linalg.norm(w_est)
    if norm_true == 0.0 or norm_est == 0.0:
        return None

    return float(np.dot(w_true, w_est) / (norm_true * norm_est))


def compute_log_likelihood(psi_array, s_array, w_est):
    """
    Average log-likelihood of preferences under logistic noise model:

      P(I | w) = 1 / (1 + exp(- I * w^T psi))

    psi_array: shape (N, D)
    s_array: shape (N,) with entries in {+1, -1, 0}
    w_est: shape (D,)

    We ignore ties (s == 0). Returns mean log P over non-tie samples.
    """
    if psi_array is None or s_array is None or w_est is None:
        return None

    psi = np.asarray(psi_array, dtype=float)
    s = np.asarray(s_array, dtype=float).reshape(-1)
    w = np.asarray(w_est, dtype=float).flatten()

    mask = (s != 0)
    if mask.sum() == 0:
        return None

    psi = psi[mask]
    s = s[mask]

    logits = s * (psi @ w)  # (N,)
    log_probs = -np.log1p(np.exp(-logits))
    return float(log_probs.mean())


class ResultsTracker:
    def __init__(self, task_name, method_name, human_topk=None, pure_oracle=False):
        self.task = task_name
        self.method = method_name
        self.human_topk = human_topk
        self.pure_oracle = pure_oracle
        self.history = []
        self.num_oracle = 0
        self.num_llm = 0
        self.start_time = time.time()


    def record_query_source(self, source):
        """
        source should ideally be:
          - "oracle"
          - "llm"

        Backward compatibility:
          - if older code still passes "human", we treat it as oracle.
        """
        if source == "oracle" or source == "human":
            self.num_oracle += 1
        elif source == "llm":
            self.num_llm += 1

    def log_iteration(self, iteration, num_queries, w_est,
                      w_true=None, psi_list=None, s_list=None):
        """
        iteration: int (1-based)
        num_queries: total comparisons seen so far (oracle + LLM)
        w_est: current mean weight vector (np.ndarray)
        w_true: ground truth weights if available (np.ndarray or None)
        psi_list: list of psi vectors up to now
        s_list: list of preference labels up to now
        """
        entry = {
            "iteration": int(iteration),
            "num_queries": int(num_queries),
            "oracle_queries": int(self.num_oracle),  # <-- key for oracle-only plotting
            "elapsed_sec": float(time.time() - self.start_time),
        }

        if w_true is not None and w_est is not None:
            entry["alignment"] = compute_alignment(w_true, w_est)

        if psi_list is not None and s_list is not None and w_est is not None:
            psi_arr = np.vstack(psi_list)
            s_arr = np.asarray(s_list)
            ll = compute_log_likelihood(psi_arr, s_arr, w_est)
            entry["avg_log_likelihood"] = ll

        self.history.append(entry)

    def to_dict(self):
        return {
            "task": self.task,
            "method": self.method,
            "human_topk": self.human_topk,
            "pure_oracle": self.pure_oracle,
            "num_oracle": int(self.num_oracle),
            "num_llm": int(self.num_llm),
            "history": self.history,
        }


def save_results(tracker, out_dir="results"):
    """
    Save all metrics to results/<task>_<method>_<timestamp>.json
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"

    if tracker.pure_oracle:
        tag = "oracle"
    else:
        tag = f"topk{tracker.human_topk}"

    fname = f"{tracker.task}_{tracker.method}_{tag}_{ts}.json"

    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(tracker.to_dict(), f, indent=2)
    print(f"[Results] Saved metrics to {path}")
    return path
