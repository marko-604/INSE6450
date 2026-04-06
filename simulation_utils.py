import scipy.optimize as opt
import algos
from models import Driver, LunarLander, MountainCar, Swimmer, Tosser, Fetch
import numpy as np


def get_feedback(simulation_object, input_A, input_B):
    simulation_object.feed(input_A)
    phi_A = simulation_object.get_features()
    simulation_object.feed(input_B)
    phi_B = simulation_object.get_features()

    psi = np.array(phi_A) - np.array(phi_B)

    s = None
    while s is None:
        selection = input('A/B to watch, 1/2/0 to vote (0 = no preference): ').lower()
        if selection == 'a':
            simulation_object.feed(input_A)
            simulation_object.watch(1)
        elif selection == 'b':
            simulation_object.feed(input_B)
            simulation_object.watch(1)
        elif selection == '1':
            s = 1
        elif selection == '2':
            s = -1
        elif selection == '0':
            s = 0

    return psi, s


def refresh_driver_w_true(env: Driver):
    base = np.array([0.35, 0.30, -0.70], dtype=float)
    if getattr(env, 'extra_feature_id', 'none') == 'none':
        w_true = base
    else:
        appended = env.hidden_extra_weight if env.extra_feature_id == env.hidden_extra_feature_id else 0.0
        w_true = np.append(base, appended)
    norm = np.linalg.norm(w_true)
    if norm > 0:
        w_true = w_true / norm
    env.w_true = w_true
    return env


def create_env(task):
    if task == 'driver':
        env = Driver()
        env.set_extra_feature('none')
        refresh_driver_w_true(env)
        return env
    elif task == 'lunarlander':
        return LunarLander()
    elif task == 'mountaincar':
        env = MountainCar()
        w_true = np.array([0.85, 0.15, -0.45], dtype=float)
        w_true /= np.linalg.norm(w_true)
        env.w_true = w_true
        return env
    elif task == 'swimmer':
        env = Swimmer()
        w_true = np.array([0.90, 0.05, -0.35], dtype=float)
        w_true /= np.linalg.norm(w_true)
        env.w_true = w_true
        return env
    elif task == 'fetch':
        env = Fetch()
        env.w_true = np.array([-0.40, -0.35, -0.10, -0.15], dtype=float)
        return env
    elif task == 'tosser':
        env = Tosser()
        def _normalize(w):
            w = np.asarray(w, dtype=float)
            n = np.linalg.norm(w)
            return w if n == 0 else w / n
        env.w_true = _normalize([0.4, 0.05, 0.0, 0.8])
        env.preference_label = "green_strong"
        return env


def run_algo(method, simulation_object, w_samples, b=10, B=200):
    if method == 'nonbatch':
        return algos.nonbatch(simulation_object, w_samples)
    if method == 'greedy':
        return algos.greedy(simulation_object, w_samples, b)
    elif method == 'medoids':
        return algos.medoids(simulation_object, w_samples, b, B)
    elif method == 'boundary_medoids':
        return algos.boundary_medoids(simulation_object, w_samples, b, B)
    elif method == 'successive_elimination':
        return algos.successive_elimination(simulation_object, w_samples, b, B)
    elif method == 'random':
        return algos.random(simulation_object, w_samples)
    elif method == 'dpp':
        return algos.dpp(simulation_object, w_samples, b, B)
    else:
        print('There is no method called ' + method)
        exit(0)


def func(ctrl_array, *args):
    simulation_object = args[0]
    w = np.array(args[1])
    simulation_object.set_ctrl(ctrl_array)
    features = simulation_object.get_features()
    return -np.mean(np.array(features).dot(w))


def perform_best(simulation_object, w, iter_count=10):
    u = simulation_object.ctrl_size
    lower_ctrl_bound = [x[0] for x in simulation_object.ctrl_bounds]
    upper_ctrl_bound = [x[1] for x in simulation_object.ctrl_bounds]
    opt_val = np.inf
    for _ in range(iter_count):
        temp_res = opt.fmin_l_bfgs_b(func, x0=np.random.uniform(low=lower_ctrl_bound, high=upper_ctrl_bound, size=(u)), args=(simulation_object, w), bounds=simulation_object.ctrl_bounds, approx_grad=True)
        if temp_res[1] < opt_val:
            optimal_ctrl = temp_res[0]
            opt_val = temp_res[1]
    simulation_object.set_ctrl(optimal_ctrl)
    keep_playing = 'y'
    while keep_playing == 'y':
        keep_playing = 'u'
        simulation_object.watch(1)
        while keep_playing != 'n' and keep_playing != 'y':
            keep_playing = input('Again? [y/n]: ').lower()
    return -opt_val
