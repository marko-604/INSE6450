from algos import generate_psi
from simulation_utils import create_env
import numpy as np
import sys
import os

task = sys.argv[1].lower()
K = int(sys.argv[2])

# optional third arg: feature id
feature_id = sys.argv[3].lower() if len(sys.argv) > 3 else "none"

simulation_object = create_env(task)

# only driver needs dynamic feature cache selection
if task == "driver" and hasattr(simulation_object, "set_extra_feature"):
    simulation_object.set_extra_feature(feature_id)

z = simulation_object.feed_size
lower_input_bound = np.array([x[0] for x in simulation_object.feed_bounds], dtype=float)
upper_input_bound = np.array([x[1] for x in simulation_object.feed_bounds], dtype=float)

pair_low = np.concatenate([lower_input_bound, lower_input_bound])
pair_high = np.concatenate([upper_input_bound, upper_input_bound])

inputs_set = np.random.uniform(
    low=pair_low,
    high=pair_high,
    size=(K, 2 * z),
)

psi_set = generate_psi(simulation_object, inputs_set)

os.makedirs("ctrl_samples", exist_ok=True)

if task == "driver":
    out_path = os.path.join("ctrl_samples", f"{simulation_object.name}_{feature_id}.npz")
else:
    out_path = os.path.join("ctrl_samples", f"{simulation_object.name}.npz")

np.savez(out_path, inputs_set=inputs_set, psi_set=psi_set)
print("Saved to:", os.path.abspath(out_path))
print("Done!")