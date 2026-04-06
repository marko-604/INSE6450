import requests
import numpy as np

from models import Driver, Tosser, Swimmer
from driver_extra_feature import EXTRA_FEATURE_BANK
from simulation_utils import get_feedback as human_get_feedback




OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"


DRIVER_HISTORY = []
BASELINE_HISTORY = []
def reset_driver_history():
    DRIVER_HISTORY.clear()
def clear_baseline_history():
    BASELINE_HISTORY.clear()
def freeze_baseline_from_current_history():
    BASELINE_HISTORY.clear()
    BASELINE_HISTORY.extend(DRIVER_HISTORY)


SWIMMER_HISTORY = []
SWIMMER_BASELINE_HISTORY = []

def reset_swimmer_history():
    SWIMMER_HISTORY.clear()

def clear_swimmer_baseline_history():
    SWIMMER_BASELINE_HISTORY.clear()

def freeze_swimmer_baseline_from_current_history():
    SWIMMER_BASELINE_HISTORY.clear()
    SWIMMER_BASELINE_HISTORY.extend(SWIMMER_HISTORY)



TOSSER_HISTORY = []
TOSSER_BASELINE_HISTORY = []
def reset_tosser_history():
    TOSSER_HISTORY.clear()
def clear_tosser_baseline_history():
    TOSSER_BASELINE_HISTORY.clear()
def freeze_tosser_baseline_from_current_history():
    TOSSER_BASELINE_HISTORY.clear()
    TOSSER_BASELINE_HISTORY.extend(TOSSER_HISTORY)



# -----------------------------
# Helpers: feature description
# -----------------------------
def describe_tosser_features(phi):
    """
    Tosser is 4D (from Tosser.get_features):
      0) horizontal_range (higher is better)
      1) maximum_altitude (higher is better)
      2) num_of_flips (higher is better)
      3) dist_to_basket (higher is better; it's exp(-3*distance))
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    lines = []
    if phi.shape[0] >= 4:
        lines.append(f"- Horizontal Range: {phi[0]:.3f}")
        lines.append(f"- Maximum Altitude: {phi[1]:.3f}")
        lines.append(f"- Number of Flips: {phi[2]:.3f}")
        lines.append(f"- Basket Proximity Score: {phi[3]:.3f}")
    return "\n".join(lines)

def _log_tosser_choice(phi_A, phi_B, s, source="llm"):
    phi_A = np.asarray(phi_A, dtype=float).reshape(-1)
    phi_B = np.asarray(phi_B, dtype=float).reshape(-1)
    psi = (phi_A - phi_B).reshape(-1)

    TOSSER_HISTORY.append(
        {"phi_A": phi_A.tolist(), "phi_B": phi_B.tolist(), "psi": psi.tolist(), "s": int(s), "source": source}
    )

def _recent_tosser_history_for_prompt(max_items: int = 6) -> str:
    history_src = TOSSER_BASELINE_HISTORY if TOSSER_BASELINE_HISTORY else TOSSER_HISTORY
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

        desc_A = describe_tosser_features(phi_A)
        desc_B = describe_tosser_features(phi_B)

        lines.append(
            f"[Past comparison {i}]\n"
            f"Trajectory A:\n{desc_A}\n\n"
            f"Trajectory B:\n{desc_B}\n"
            f"Decision: {decision}\n"
        )

    return "\n---\n".join(lines) if lines else "No feature history available."


def describe_driver_features(phi):
    """
    Driver base is now 3D:
      0) staying in lane
      1) heading straightness
      2) collision avoidance cost
    If extended to 4D, the last value is the currently active extra feature.
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    lines = []
    if phi.shape[0] >= 3:
        lines.append(f"- Staying In Lane: {phi[0]:.3f}")
        lines.append(f"- Heading Straightness: {phi[1]:.3f}")
        lines.append(f"- Collision Avoidance Cost: {phi[2]:.3f}")
    if phi.shape[0] >= 4:
        lines.append(f"- Active Extra Feature: {phi[3]:.3f}")
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

def _recent_driver_history_all_for_prompt(max_items: int = 8) -> str:
    if not DRIVER_HISTORY:
        return "No previous comparisons yet."

    hist = DRIVER_HISTORY[-max_items:]
    lines = []
    start_idx = max(1, len(DRIVER_HISTORY) - max_items + 1)

    for i, entry in enumerate(hist, start=start_idx):
        phi_A = entry.get("phi_A")
        phi_B = entry.get("phi_B")
        s = entry.get("s", 0)
        source = entry.get("source", "unknown")

        if s == 1:
            decision = f"{source.upper()} preferred Trajectory A."
        elif s == -1:
            decision = f"{source.upper()} preferred Trajectory B."
        else:
            decision = f"{source.upper()} had no preference (tie)."

        if phi_A is None or phi_B is None:
            continue

        desc_A = describe_driver_features(phi_A)
        desc_B = describe_driver_features(phi_B)

        lines.append(
            f"[Recent comparison {i}]\n"
            f"Trajectory A:\n{desc_A}\n\n"
            f"Trajectory B:\n{desc_B}\n"
            f"Decision: {decision}\n"
        )

    return "\n---\n".join(lines) if lines else "No feature history available."


def llm_choose_driver_extra_feature(simulation_object) -> str:
    current_feature = getattr(simulation_object, "extra_feature_id", "none")
    history_str = _recent_driver_history_all_for_prompt(max_items=8)

    options = []
    for feature_id, meta in EXTRA_FEATURE_BANK.items():
        if feature_id == "none":
            continue
        options.append(f"- {feature_id}: {meta['description']}")

    prompt = f"""
You are deciding whether the driver reward model is missing one important extra feature.

Fixed driver features already present:
1. Staying in lane (higher is better).
2. Heading straightness (higher is better).
3. Collision avoidance cost (lower is better).

Current active extra feature: {current_feature}

Candidate extra features:
{chr(10).join(options)}

Recent preference evidence:
{history_str}

Choose exactly one feature id from the candidate list above if one extra feature should be active.
If none of them seems necessary right now, answer exactly: none

Return exactly one token: the feature id only.
"""

    try:
        raw = call_ollama_mistral(prompt).strip().lower()
    except Exception as e:
        print(f"[WARN] LLM feature-selection call failed ({type(e).__name__}): {e} -> keeping current feature")
        return current_feature

    if raw == "none":
        return "none"
    if raw in EXTRA_FEATURE_BANK and raw != "none":
        return raw

    print(f"[WARN] LLM returned unknown feature '{raw}', keeping current feature '{current_feature}'.")
    return current_feature


def describe_swimmer_features(phi):
    """
    Swimmer is 3D (from simulation_utils.create_env comments):
      0) horizontal_range     (higher is better)
      1) vertical_range       (closer to zero is better / mostly irrelevant)
      2) total_displacement   (lower is better; avoid wasted motion)
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    lines = []
    if phi.shape[0] >= 3:
        lines.append(f"- Forward Progress: {phi[0]:.3f}")
        lines.append(f"- Vertical Drift: {phi[1]:.3f}")
        lines.append(f"- Total Displacement / Effort: {phi[2]:.3f}")
    return "\n".join(lines)


def _log_swimmer_choice(phi_A, phi_B, s, source="llm"):
    phi_A = np.asarray(phi_A, dtype=float).reshape(-1)
    phi_B = np.asarray(phi_B, dtype=float).reshape(-1)
    psi = (phi_A - phi_B).reshape(-1)

    SWIMMER_HISTORY.append(
        {"phi_A": phi_A.tolist(), "phi_B": phi_B.tolist(), "psi": psi.tolist(), "s": int(s), "source": source}
    )


def _recent_swimmer_history_for_prompt(max_items: int = 6) -> str:
    history_src = SWIMMER_BASELINE_HISTORY if SWIMMER_BASELINE_HISTORY else SWIMMER_HISTORY
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

        desc_A = describe_swimmer_features(phi_A)
        desc_B = describe_swimmer_features(phi_B)

        lines.append(
            f"[Past comparison {i}]\n"
            f"Trajectory A:\n{desc_A}\n\n"
            f"Trajectory B:\n{desc_B}\n"
            f"Decision: {decision}\n"
        )

    return "\n---\n".join(lines) if lines else "No feature history available."

def llm_preference_for_swimmer(simulation_object, input_A, input_B):
    simulation_object.feed(input_A)
    phi_A = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

    simulation_object.feed(input_B)
    phi_B = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

    psi = (phi_A - phi_B).reshape(-1)

    history_str = _recent_swimmer_history_for_prompt(max_items=6)

    prompt = f"""
You are labeling pairwise preferences between two swimmer trajectories.

Prefer trajectories that:
- make more forward progress,
- avoid unnecessary vertical drift,
- use less wasted motion / total displacement.

Previous comparisons (context):
{history_str}

Now evaluate:

Trajectory A:
{describe_swimmer_features(phi_A)}

Trajectory B:
{describe_swimmer_features(phi_B)}

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

    if isinstance(simulation_object, Swimmer) and not SWIMMER_BASELINE_HISTORY:
        _log_swimmer_choice(phi_A=phi_A, phi_B=phi_B, s=s, source="llm")

    return np.asarray(psi, dtype=float), int(s)




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




def llm_preference_for_tosser(simulation_object, input_A, input_B):
    simulation_object.feed(input_A)
    phi_A = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

    simulation_object.feed(input_B)
    phi_B = np.asarray(simulation_object.get_features(), dtype=float).reshape(-1)

    psi = (phi_A - phi_B).reshape(-1)

    history_str = _recent_tosser_history_for_prompt(max_items=6)

    prompt = f"""
You are labeling pairwise preferences between two tossing trajectories.
Prefer trajectories that:
- throw farther (higher range),
- go higher (higher altitude),
- flip more (higher flips),
- land closer to a basket (higher basket proximity score).

Previous comparisons (context):
{history_str}

Now evaluate:

Trajectory A:
{describe_tosser_features(phi_A)}

Trajectory B:
{describe_tosser_features(phi_B)}

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

    # mirror driver behavior: log only until baseline exists
    if isinstance(simulation_object, Tosser) and not TOSSER_BASELINE_HISTORY:
        _log_tosser_choice(phi_A=phi_A, phi_B=phi_B, s=s, source="llm")

    return np.asarray(psi, dtype=float), int(s)





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

    active_extra = getattr(simulation_object, "extra_feature_id", "none")

    prompt = f"""
You are labeling pairwise preferences between two driving trajectories and your goal is to mimic that of a safe human.

The driver always has these fixed features:
1. Staying in lane (higher is better).
2. Heading straightness (higher is better).
3. Collision avoidance cost (lower is better).

Current active extra feature: {active_extra}
If the feature vector has a 4th value, that last value is the currently active extra feature.

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
            elif isinstance(simulation_object, Tosser):
                _log_tosser_choice(phi_A=phi_A, phi_B=phi_B, s=s, source="oracle")
            elif isinstance(simulation_object, Swimmer):
                _log_swimmer_choice(phi_A=phi_A, phi_B=phi_B, s=s, source="oracle")


        else:
            psi, s = _human_feedback_psi(simulation_object, input_A, input_B)
            source = "human"
            print(f"[INFO] Driver query {query_index + 1}/{batch_size}: FORCED HUMAN")

        if tracker:
            tracker.record_query_source(source)
        return np.asarray(psi, dtype=float), int(s), source


    if routing_mode == "topk":
        if isinstance(simulation_object, Tosser) or task == "tosser":
            psi, s = llm_preference_for_tosser(simulation_object, input_A, input_B)
        elif isinstance(simulation_object, Swimmer) or task == "swimmer":
            psi, s = llm_preference_for_swimmer(simulation_object, input_A, input_B)
        else:
            psi, s = llm_preference_for_driver(simulation_object, input_A, input_B)

        if tracker:
            tracker.record_query_source("llm")
        print(f"[INFO] Driver query {query_index + 1}/{batch_size}: TOPK MODE -> LLM")
        return np.asarray(psi, dtype=float), int(s), "llm"

    if isinstance(simulation_object, Tosser) or task == "tosser":
        psi, s = llm_preference_for_tosser(simulation_object, input_A, input_B)
    elif isinstance(simulation_object, Swimmer) or task == "swimmer":
        psi, s = llm_preference_for_swimmer(simulation_object, input_A, input_B)
    else:
        psi, s = llm_preference_for_driver(simulation_object, input_A, input_B)

    if tracker:
        tracker.record_query_source("llm")
    return np.asarray(psi, dtype=float), int(s), "llm"
