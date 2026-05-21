import sys
import os
import time
import numpy as np
import torch
from pyqubo import Binary
from neal import SimulatedAnnealingSampler

Num_reads = 10
beta_min = 0.1
beta_max = 4.0

data_root = sys.argv[1]
data_folder = sys.argv[2]
data_id = sys.argv[3]
seed = int(sys.argv[4])

dev = "cpu"

def load_Q(filename):
    Q_np = np.load(filename)
    return torch.tensor(Q_np, dtype=torch.float32, device=dev)

def convert_Q_to_pyqubo(Q):
    n = Q.shape[0]
    qubo_dict = {}
    x = {idx: Binary(f"x{idx}") for idx in range(n)}

    for row in range(n):
        for col in range(row, n):
            qubo_dict[(f"x{row}", f"x{col}")] = Q[row, col].item()

    return qubo_dict, x

def solve_with_pyqubo(Q):
    qubo_dict, x = convert_Q_to_pyqubo(Q)

    H = sum(
        value * x[int(key[0][1:])] * x[int(key[1][1:])]
        for key, value in qubo_dict.items()
    )

    model = H.compile()
    qubo, _ = model.to_qubo()

    sampler = SimulatedAnnealingSampler()

    start_time = time.time()

    sampleset = sampler.sample_qubo(
        qubo,
        num_reads=Num_reads,
        beta_range=(beta_min, beta_max),
        beta_schedule_type="geometric",
        seed=seed
    )

    elapsed_time = time.time() - start_time

    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy

    pyqubo_solution = torch.tensor(
        [int(best_sample.get(f"x{idx}", 0)) for idx in range(len(x))],
        dtype=torch.int
    )

    return pyqubo_solution, best_energy, elapsed_time

Q_filename = os.path.join(data_root, data_folder, f"data_{data_id}.npy")

result_folder = os.path.join("solution", data_folder, f"data_{data_id}")
solution_folder = os.path.join("solutionX", data_folder, f"data_{data_id}")

os.makedirs(result_folder, exist_ok=True)
os.makedirs(solution_folder, exist_ok=True)

result_filename = os.path.join(result_folder, f"pyqubo_{seed}")
solution_filename = os.path.join(solution_folder, f"pyqubo_{seed}")

Q = load_Q(Q_filename)
pyqubo_solution, pyqubo_energy, pyqubo_time = solve_with_pyqubo(Q)

with open(solution_filename, "w") as f:
    f.write("PyQUBO Solution:\n")
    f.write(" ".join(map(str, pyqubo_solution.tolist())) + "\n")

with open(result_filename, "w") as f:
    f.write(f"PyQUBO Energy: {pyqubo_energy}\n")
    f.write(f"PyQUBO Time: {pyqubo_time:.6f} sec\n")

