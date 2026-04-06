import numpy as np
import scipy.optimize as opt
from sklearn.metrics.pairwise import pairwise_distances
import kmedoids
from scipy.spatial import ConvexHull
import dpp_sampler
import os
import subprocess
import sys

def _ctrl_sample_path(simulation_object):
    if getattr(simulation_object, "name", "") == "driver":
        feature_id = getattr(simulation_object, "extra_feature_id", "none")
        return os.path.join("ctrl_samples", f"{simulation_object.name}_{feature_id}.npz")
    return os.path.join("ctrl_samples", f"{simulation_object.name}.npz")


def _ensure_ctrl_samples(simulation_object, K=4000):
    path = _ctrl_sample_path(simulation_object)
    if os.path.exists(path):
        return path

    os.makedirs("ctrl_samples", exist_ok=True)

    task = simulation_object.name.lower()
    if task == "driver":
        feature_id = getattr(simulation_object, "extra_feature_id", "none")
        cmd = [sys.executable, "input_sampler_2.py", task, str(K), feature_id]
    else:
        cmd = [sys.executable, "input_sampler_2.py", task, str(K)]

    print(f"[ctrl_samples] generating missing cache: {path}")
    subprocess.run(cmd, check=True)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected ctrl sample file was not created: {path}")

    return path

def func_psi(psi_set, w_samples):
    y = psi_set.dot(w_samples.T)
    term1 = np.sum(1.-np.exp(-np.maximum(y,0)),axis=1)
    term2 = np.sum(1.-np.exp(-np.maximum(-y,0)),axis=1)
    f = -np.minimum(term1,term2)
    return f

def generate_psi(simulation_object, inputs_set):
    z = simulation_object.feed_size
    inputs_set = np.array(inputs_set)
    if len(inputs_set.shape) == 1:
        inputs1 = inputs_set[0:z].reshape(1,z)
        inputs2 = inputs_set[z:2*z].reshape(1,z)
        input_count = 1
    else:
        inputs1 = inputs_set[:,0:z]
        inputs2 = inputs_set[:,z:2*z]
        input_count = inputs_set.shape[0]
    d = simulation_object.num_of_features
    features1 = np.zeros([input_count, d])
    features2 = np.zeros([input_count, d])  
    for i in range(input_count):
        simulation_object.feed(list(inputs1[i]))
        features1[i] = simulation_object.get_features()
        simulation_object.feed(list(inputs2[i]))
        features2[i] = simulation_object.get_features()
    psi_set = features1 - features2
    return psi_set

def func(inputs_set, *args):
    simulation_object = args[0]
    w_samples = args[1]
    psi_set = generate_psi(simulation_object, inputs_set)
    return func_psi(psi_set, w_samples)

def nonbatch(simulation_object, w_samples):
    z = simulation_object.feed_size
    lower_input_bound = np.array([x[0] for x in simulation_object.feed_bounds], dtype=float)
    upper_input_bound = np.array([x[1] for x in simulation_object.feed_bounds], dtype=float)

    pair_low = np.concatenate([lower_input_bound, lower_input_bound])
    pair_high = np.concatenate([upper_input_bound, upper_input_bound])

    opt_res = opt.fmin_l_bfgs_b(
        func,
        x0=np.random.uniform(low=pair_low, high=pair_high, size=(2 * z)),
        args=(simulation_object, w_samples),
        bounds=simulation_object.feed_bounds * 2,
        approx_grad=True,
    )
    return opt_res[0][0:z], opt_res[0][z:2*z]


def select_top_candidates(simulation_object, w_samples, B):
    d = simulation_object.num_of_features
    z = simulation_object.feed_size

    inputs_set = np.zeros(shape=(0,2*z))
    psi_set = np.zeros(shape=(0,d))
    f_values = np.zeros(shape=(0))
    path = _ensure_ctrl_samples(simulation_object)
    data = np.load(path)
    inputs_set = data['inputs_set']
    psi_set = data['psi_set']
    f_values = func_psi(psi_set, w_samples)
    id_input = np.argsort(f_values)
    inputs_set = inputs_set[id_input[0:B]]
    psi_set = psi_set[id_input[0:B]]
    f_values = f_values[id_input[0:B]]
    return inputs_set, psi_set, f_values, d, z

def greedy(simulation_object, w_samples, b):
    inputs_set, _, _, _, z = select_top_candidates(simulation_object, w_samples, b)
    return inputs_set[:, :z], inputs_set[:, z:]

def medoids(simulation_object, w_samples, b, B=200):
    inputs_set, psi_set, _, _, z = select_top_candidates(simulation_object, w_samples, B)

    D = pairwise_distances(psi_set, metric='euclidean')
    M, C = kmedoids.kMedoids(D, b)
    return inputs_set[M, :z], inputs_set[M, z:]

def boundary_medoids(simulation_object, w_samples, b, B=200):
    inputs_set, psi_set, _, _, z = select_top_candidates(simulation_object, w_samples, B)

    hull = ConvexHull(psi_set)
    simplices = np.unique(hull.simplices)
    boundary_psi = psi_set[simplices]
    boundary_inputs = inputs_set[simplices]
    D = pairwise_distances(boundary_psi, metric='euclidean')
    M, C = kmedoids.kMedoids(D, b)
    
    return boundary_inputs[M, :z], boundary_inputs[M, z:]

def successive_elimination(simulation_object, w_samples, b, B=200):
    inputs_set, psi_set, f_values, d, z = select_top_candidates(simulation_object, w_samples, B)

    D = pairwise_distances(psi_set, metric='euclidean')
    D = np.array([np.inf if x==0 else x for x in D.reshape(B*B,1)]).reshape(B,B)
    while len(inputs_set) > b:
        ij_min = np.where(D == np.min(D))
        if len(ij_min) > 1 and len(ij_min[0]) > 1:
            ij_min = ij_min[0]
        elif len(ij_min) > 1:
            ij_min = np.array([ij_min[0],ij_min[1]])

        if f_values[ij_min[0]] < f_values[ij_min[1]]:
            delete_id = ij_min[1]
        else:
            delete_id = ij_min[0]
        D = np.delete(D, delete_id, axis=0)
        D = np.delete(D, delete_id, axis=1)
        f_values = np.delete(f_values, delete_id)
        inputs_set = np.delete(inputs_set, delete_id, axis=0)
        psi_set = np.delete(psi_set, delete_id, axis=0)
    return inputs_set[:,0:z], inputs_set[:,z:2*z]

def dpp(simulation_object, w_samples, b, B=200, gamma=1):
    inputs_set, psi_set, f_values, _, z = select_top_candidates(simulation_object, w_samples, B)

    ids = dpp_sampler.sample_ids_mc(psi_set, -f_values, b, alpha=4, gamma=gamma, steps=0) # alpha is not important because it is greedy-dpp
    return inputs_set[ids,:z], inputs_set[ids,z:]

def random(simulation_object, w_samples):
    lower_input_bound = np.array([x[0] for x in simulation_object.feed_bounds], dtype=float)
    upper_input_bound = np.array([x[1] for x in simulation_object.feed_bounds], dtype=float)

    pair_low = np.concatenate([lower_input_bound, lower_input_bound])
    pair_high = np.concatenate([upper_input_bound, upper_input_bound])

    input_A = np.random.uniform(low=pair_low, high=pair_high, size=(2 * simulation_object.feed_size))
    input_B = np.random.uniform(low=pair_low, high=pair_high, size=(2 * simulation_object.feed_size))
    return input_A, input_B