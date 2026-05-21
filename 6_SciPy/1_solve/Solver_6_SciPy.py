import sys
import os
import time
import numpy as np
from scipy.optimize import minimize

slop = 0.5
Num_Step = 1000000

data_root = sys.argv[1]
data_folder = sys.argv[2]
threshold_name = sys.argv[3]
data_id = sys.argv[4]
seed = int(sys.argv[5])

threshold = float(threshold_name)

def sigmoid_scaled(x):
    return 1.0 / (1.0 + np.exp(-slop * (x - 0.5)))

def loss_and_grad(x, upper_Q, grad_Q):
    z = sigmoid_scaled(x)

    loss = float(z @ upper_Q @ z)

    grad_z = grad_Q @ z
    dz_dx = slop * z * (1.0 - z)
    grad_x = grad_z * dz_dx

    return loss, grad_x

def solve_with_scipy(Q, threshold, seed):
    rng = np.random.default_rng(seed)

    upper_Q = np.triu(Q)
    grad_Q = upper_Q + upper_Q.T

    n = Q.shape[0]
    x0 = rng.standard_normal(n)
    bounds = [(-5.0, 5.0)] * n

    start_time = time.time()

    res = minimize(
        loss_and_grad,
        x0,
        args=(upper_Q, grad_Q),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": Num_Step,
            "ftol": threshold
        }
    )

    elapsed_time = time.time() - start_time

    x_final = res.x
    x_bin = (sigmoid_scaled(x_final) > 0.5).astype(float)
    energy = float(x_bin @ upper_Q @ x_bin.T)

    step_num = res.nit

    return x_bin.astype(int), energy, elapsed_time, step_num

Q_filename = os.path.join(data_root, data_folder, f"data_{data_id}.npy")

result_folder = os.path.join(
    "solution",
    data_folder,
    f"thr_{threshold_name}",
    f"data_{data_id}"
)

solution_folder = os.path.join(
    "solutionX",
    data_folder,
    f"thr_{threshold_name}",
    f"data_{data_id}"
)

os.makedirs(result_folder, exist_ok=True)
os.makedirs(solution_folder, exist_ok=True)

result_filename = os.path.join(result_folder, f"scipy_{seed}")
solution_filename = os.path.join(solution_folder, f"scipy_{seed}")

Q = np.load(Q_filename).astype(np.float64, copy=False)

solution, energy, time_used, step_num = solve_with_scipy(Q, threshold, seed)

with open(result_filename, "w") as f:
    f.write(f"SciPy Energy: {energy}\n")
    f.write(f"SciPy Time: {time_used:.6f} sec\n")
    f.write(f"SciPy Steps: {step_num}\n")

with open(solution_filename, "w") as f:
    f.write("SciPy Solution:\n")
    f.write(" ".join(map(str, solution.tolist())) + "\n")

