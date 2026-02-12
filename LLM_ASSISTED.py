import requests
import numpy as np

from models import Driver
from simulation_utils import get_feedback as human_get_feedback




OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


# Rolling history (normally would grow, but we will freeze baseline + stop logging after that)
DRIVER_HISTORY = []

# Frozen baseline history (only this is used for LLM prompt context after it exists)
BASELINE_HISTORY = []






def reset_driver_history():
    DRIVER_HISTORY.clear()


def clear_baseline_history():
    BASELINE_HISTORY.clear()


def freeze_baseline_from_current_history():
    BASELINE_HISTORY.clear()
    BASELINE_HISTORY.extend(DRIVER_HISTORY)


# -----------------------------
# Helpers: feature description
# -----------------------------
def describe_driver_features(phi):
    """
    Default driver is 4D: [lane_center, collision_avoid, road_pref, speed]
    If extended to 5D, last dimension is "extra feature".
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    lines = []
    if phi.shape[0] >= 4:
        lines.append(f"- Lane Centering: {phi[0]:.3f}")
        lines.append(f"- Collision Avoidance: {phi[1]:.3f}")
        lines.append(f"- Road Preference: {phi[2]:.3f}")
        lines.append(f"- Speed: {phi[3]:.3f}")
    if phi.shape[0] >= 5:
        lines.append(f"- Extra Feature: {phi[4]:.3f}")
    return "\n".join(lines)


def _log_driver_choice(phi_A, phi_B, s, source="llm"):
    phi_A = np.asarray(phi_A, dtype=float).reshape(-1)
    phi_B = np.asarray(phi_B, dtype=float).reshape(-1)
    psi = (phi_A - phi_B).reshape(-1)

    DRIVER_HISTORY.append(
        {
            "phi_A": phi_A.tolist(),
            "phi_B": phi_B.tolist(),
            "psi": psi.tolist(),
            "s": int(s),
            "source": source,
        }
    )


def _recent_feature_history_for_prompt(max_items: int = 6) -> str:
    """
    If BASELINE_HISTORY exists, use ONLY it. Otherwise use DRIVER_HISTORY.
    """
    history_src = BASELINE_HISTORY if BASELINE_HISTORY else DRIVER_HISTORY

    if not history_src:
        return "No previous comparisons yet."

    hist = history_src[-max_items:]
    lines = []

    start_idx = max(1, len(history_src) - max_items + 1)
    for i, entry in enumerate(hist, start=start_idx):
        phi_A = entry.get("phi_A")
        phi_B = entry.get("phi_B")
        s = entry.get("s", 0)

        if s == 1:
            decision = "Oracle preferred Trajectory A."
        elif s == -1:
            decision = "Oracle preferred Trajectory B."
        else:
            decision = "Oracle had no preference (tie)."



        if phi_A is None or phi_B is None:
            continue

        desc_A = describe_driver_features(phi_A)
        desc_B = describe_driver_features(phi_B)

        lines.append(
            f"[Past comparison {i}]\n"
            f"Trajectory A:\n{desc_A}\n\n"
            f"Trajectory B:\n{desc_B}\n"
            f"Decision: {decision}\n"
        )

    return "\n---\n".join(lines) if lines else "No feature history available."


# -----------------------------
# Feedback sources
# -----------------------------
def _human_feedback_psi(simulation_object, input_A, input_B):
    result = human_get_feedback(simulation_object, input_A, input_B)
    if isinstance(result, tuple) and len(result) >= 2:
        psi, s = result[0], result[1]
    else:
        raise RuntimeError("Unexpected return format from human_get_feedback.")
    return np.asarray(psi, dtype=float), int(s)


def oracle_preference_for_driver(simulation_object, input_A, input_B, tie_eps=1e-9):
    """
    Oracle preference using env ground-truth weights w_true.
    Returns: (phi_A, phi_B, psi, s)
    """
    if not hasattr(simulation_object, "w_true") or simulation_object.w_true is None:
        raise RuntimeError("Oracle requested but simulation_object.w_true is missing.")

    simulation_object.feed(input_A)
    phi_A = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

    simulation_object.feed(input_B)
    phi_B = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

    psi = (phi_A - phi_B).reshape(-1)

    w_true = np.asarray(simulation_object.w_true, dtype=float).reshape(-1)
    if w_true.shape[0] != psi.shape[0]:
        raise RuntimeError(
            f"Oracle dim mismatch: w_true has {w_true.shape[0]} dims, psi has {psi.shape[0]} dims."
        )

    score = float(w_true @ psi)
    if abs(score) <= tie_eps:
        s = 0
    elif score > 0:
        s = 1
    else:
        s = -1

    return phi_A, phi_B, np.asarray(psi, dtype=float), int(s)


# -----------------------------
# LLM
# -----------------------------
def call_ollama_mistral(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def parse_preference(text: str) -> int:
    t = (text or "").strip().lower()
    if t in {"1", "a"}:
        return 1
    if t in {"2", "b"}:
        return -1
    if t in {"0", "tie", "equal"}:
        return 0
    return 0


def llm_preference_for_driver(simulation_object, input_A, input_B):
    """
    Returns: (psi, s)
    """
    simulation_object.feed(input_A)
    phi_A = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

    simulation_object.feed(input_B)
    phi_B = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

    psi = (phi_A - phi_B).reshape(-1)

    history_str = _recent_feature_history_for_prompt(max_items=6)

    prompt = f"""
You are labeling pairwise preferences between two driving trajectories and your goal is mimic that of a safe human.

   The driver has four features:
    1. Staying in lane (higher is better).
    2. Speed deviation from the desired speed (lower is better).
    3. Heading straightness (higher is better).
    4. Collision risk (lower is better).


Previous comparisons (context):
{history_str}

Now evaluate:

Trajectory A:
{describe_driver_features(phi_A)}

Trajectory B:
{describe_driver_features(phi_B)}

Answer with exactly ONE token:
1  (prefer A)
2  (prefer B)
0  (tie / no preference)
"""

    try:
        raw = call_ollama_mistral(prompt)
        s = parse_preference(raw)
    except Exception as e:
        print(f"[WARN] LLM call failed ({type(e).__name__}): {e} -> returning tie")
        s = 0

    # IMPORTANT: after baseline exists, we do NOT add more to history
    if isinstance(simulation_object, Driver) and not BASELINE_HISTORY:
        _log_driver_choice(phi_A=phi_A, phi_B=phi_B, s=s, source="llm")

    return np.asarray(psi, dtype=float), int(s)


# -----------------------------
# Routing metric (kept for your debug prints)
# -----------------------------
def predictive_entropy(w_samples: np.ndarray, psi: np.ndarray) -> float:
    """
    Posterior predictive entropy.
    Max is ln(2)=0.693...
    """
    psi = np.asarray(psi, dtype=float).reshape(-1)
    W = np.asarray(w_samples, dtype=float)
    logits = W @ psi
    probs = 1.0 / (1.0 + np.exp(-logits))
    p = float(np.mean(probs))
    p = max(min(p, 1.0 - 1e-9), 1e-9)
    H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(H)


# -----------------------------
# MAIN ENTRY: mixed feedback
# -----------------------------
def get_feedback_mixed(
    task,
    simulation_object,
    input_A,
    input_B,
    query_index,
    batch_size,
    tracker=None,
    w_samples=None,
    force_human=False,
    routing_mode="topk",
    oracle=True,
    entropy_threshold=0.55,
    log_to_history=False,
):
    """
    You asked for:
      - baseline batch: forced oracle, and store comparisons for LLM history
      - after baseline: pure LLM, no interactive human, no oracle

    We implement it by:
      - when force_human=True and oracle=True: use oracle
      - if log_to_history=True: store (phi_A, phi_B, s) into DRIVER_HISTORY
      - when BASELINE_HISTORY exists, LLM uses only baseline history
    """
    if force_human:
        if oracle:
            phi_A, phi_B, psi, s = oracle_preference_for_driver(simulation_object, input_A, input_B)
            source = "oracle"
            print(f"[INFO] Driver query {query_index + 1}/{batch_size}: FORCED ORACLE")

            # Log ALL oracle-labelled comparisons into the prompt history
            # (This becomes the running context used in "Previous comparisons")
            if isinstance(simulation_object, Driver):
                _log_driver_choice(phi_A=phi_A, phi_B=phi_B, s=s, source="oracle")

        else:
            psi, s = _human_feedback_psi(simulation_object, input_A, input_B)
            source = "human"
            print(f"[INFO] Driver query {query_index + 1}/{batch_size}: FORCED HUMAN")

        if tracker:
            tracker.record_query_source(source)
        return np.asarray(psi, dtype=float), int(s), source

    # Pure LLM path (what you want after baseline)
    if routing_mode == "topk":
        psi, s = llm_preference_for_driver(simulation_object, input_A, input_B)
        if tracker:
            tracker.record_query_source("llm")
        print(f"[INFO] Driver query {query_index + 1}/{batch_size}: TOPK MODE -> LLM")
        return np.asarray(psi, dtype=float), int(s), "llm"

    # If you ever use entropy routing again, you can re-add it here.
    psi, s = llm_preference_for_driver(simulation_object, input_A, input_B)
    if tracker:
        tracker.record_query_source("llm")
    return np.asarray(psi, dtype=float), int(s), "llm"
