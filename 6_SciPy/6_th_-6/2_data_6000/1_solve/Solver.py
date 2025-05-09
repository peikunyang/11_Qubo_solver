import sys
import time
import numpy as np
from scipy.optimize import minimize
import os

Num = 6000
slop = 0.5
threshold = 1e-6
Num_Step = 1000000

def sigmoid_scaled(x):
    return 1 / (1 + np.exp(-slop * (x - 0.5)))

def loss_fn(x, Q):
    x2 = sigmoid_scaled(x).reshape(1, -1)
    return float((x2 @ Q @ x2.T).squeeze())

def solve_with_scipy(Q):
    start_time = time.time()
    n = Q.shape[0]
    x0 = np.random.randn(n)

    bounds = [(-5.0, 5.0)] * n
    res = minimize(loss_fn, x0, args=(Q,), method='L-BFGS-B', bounds=bounds, options={
        'maxiter': Num_Step,
        'ftol': threshold
    })

    x_final = res.x
    x_bin = (sigmoid_scaled(x_final) > 0.5).astype(float)
    upper_Q = np.triu(Q)
    energy = float(x_bin @ upper_Q @ x_bin.T)

    elapsed_time = time.time() - start_time
    step_num = res.nit  # 加上這行

    return x_bin.astype(int), energy, elapsed_time, step_num

if __name__ == "__main__":
    i = sys.argv[1]
    j = sys.argv[2]
    Q_filename = f"../../../../1_gen_coe/2_data_{Num}/data_{i}.npy"
    result_filename = f"solution/{i}/scipy_{j}"
    solution_filename = f"solutionX/{i}/scipy_{j}"

    os.makedirs(f"solution/{i}", exist_ok=True)
    os.makedirs(f"solutionX/{i}", exist_ok=True)

    Q = np.load(Q_filename)
    solution, energy, time_used, step_num = solve_with_scipy(Q)

    with open(result_filename, "w") as f:
      f.write(f"SciPy Energy: {energy}\n")
      f.write(f"SciPy Time: {time_used:.6f} sec\n")
      f.write(f"SciPy Steps: {step_num}\n")  # 修正名稱為 SciPy

    with open(solution_filename, "w") as f:
        f.write("SciPy Solution:\n")
        f.write(" ".join(map(str, solution.tolist())) + "\n")

