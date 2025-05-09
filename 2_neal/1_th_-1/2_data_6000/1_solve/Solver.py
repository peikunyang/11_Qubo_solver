import sys
import os
import time
import numpy as np
import torch
from pyqubo import Binary
from neal import SimulatedAnnealingSampler

Num = 6000
Num_reads = 10
beta_min = 0.1
beta_max = 4.0
i = sys.argv[1]
j = sys.argv[2]

dev = 'cpu'

def load_Q(filename):
    Q_np = np.load(filename)
    return torch.tensor(Q_np, dtype=torch.float32, device=dev)

def convert_Q_to_pyqubo(Q):
    n = Q.shape[0]
    qubo_dict = {}
    x = {i: Binary(f'x{i}') for i in range(n)}
    for i in range(n):
        for j in range(i, n):
            qubo_dict[(f'x{i}', f'x{j}')] = Q[i, j].item()
    return qubo_dict, x

def solve_with_pyqubo(Q):
    start_time = time.time()
    qubo_dict, x = convert_Q_to_pyqubo(Q)

    H = sum(value * x[int(key[0][1:])] * x[int(key[1][1:])] for key, value in qubo_dict.items())

    model = H.compile()
    qubo, _ = model.to_qubo()

    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(
        qubo,
        num_reads=Num_reads,
        beta_range=(beta_min, beta_max),
        beta_schedule_type="geometric"
    )

    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy

    pyqubo_solution = torch.tensor([int(best_sample.get(f'x{i}', 0)) for i in range(len(x))], dtype=torch.int)

    elapsed_time = time.time() - start_time
    return pyqubo_solution, best_energy, elapsed_time

Q_filename = f"../../../../1_gen_coe/2_data_{Num}/data_{i}.npy"

result_filename1 = f"solutionX/{i}/pyqubo_{j}"
result_filename2 = f"solution/{i}/pyqubo_{j}"

Q = load_Q(Q_filename)
pyqubo_solution, pyqubo_energy, pyqubo_time = solve_with_pyqubo(Q)

with open(result_filename1, "w") as f:
    f.write("PyQUBO Solution:\n")
    f.write(" ".join(map(str, pyqubo_solution.tolist())) + "\n")

with open(result_filename2, "w") as f:
    f.write(f"PyQUBO Energy: {pyqubo_energy}\nPyQUBO Time: {pyqubo_time:.6f} sec\n")

