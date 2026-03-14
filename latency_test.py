import time
import numpy as np
from simulation_utils import create_env
from algos import generate_psi


# ------------------------------------------------
# corruption function (same idea as stress tests)
# ------------------------------------------------

def corrupt_features(psi, noise_std=0.05, mask_prob=0.2):

    psi_corrupt = psi.copy()

    # gaussian noise
    noise = np.random.normal(0, noise_std, size=psi_corrupt.shape)
    psi_corrupt += noise

    # random masking
    mask = np.random.rand(*psi_corrupt.shape) < mask_prob
    psi_corrupt[mask] = 0.0

    return psi_corrupt


# ------------------------------------------------
# setup
# ------------------------------------------------

task = "driver"
env = create_env(task)

data = np.load(f"ctrl_samples/{task}.npz")
inputs = data["inputs_set"][:500]


# ------------------------------------------------
# CLEAN LATENCY
# ------------------------------------------------

start_clean = time.time()

psi_clean = generate_psi(env, inputs)

end_clean = time.time()

clean_time = end_clean - start_clean
clean_latency = clean_time / len(inputs)
clean_throughput = len(inputs) / clean_time


# ------------------------------------------------
# CORRUPTED LATENCY
# ------------------------------------------------

start_corrupt = time.time()

psi_clean = generate_psi(env, inputs)
psi_corrupt = corrupt_features(psi_clean)

end_corrupt = time.time()

corrupt_time = end_corrupt - start_corrupt
corrupt_latency = corrupt_time / len(inputs)
corrupt_throughput = len(inputs) / corrupt_time


# ------------------------------------------------
# results
# ------------------------------------------------

print("\n--- CLEAN INPUTS ---")
print("Total time:", clean_time)
print("Latency per query:", clean_latency)
print("Throughput (queries/sec):", clean_throughput)

print("\n--- CORRUPTED INPUTS ---")
print("Total time:", corrupt_time)
print("Latency per query:", corrupt_latency)
print("Throughput (queries/sec):", corrupt_throughput)