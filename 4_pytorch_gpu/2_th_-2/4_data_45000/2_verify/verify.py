import sys
import torch
import numpy as np

Num = 45000
i = sys.argv[1]
j = sys.argv[2]

def load_Q(filename):
    Q_np = np.load(filename)
    return torch.tensor(Q_np, dtype=torch.float)

def load_solution(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        solution = list(map(int, lines[1].strip().split()))
    solution_tensor = torch.tensor(solution, dtype=torch.int)
    return solution_tensor

def verify_solution(Q, solution):
    upper_Q = torch.triu(Q)
    return (solution.to(Q.dtype) @ upper_Q @ solution.to(Q.dtype)).item()

Q = load_Q(f"../../../../1_gen_coe/4_data_{Num}/data_{i}.npy")
solution = load_solution(f"../1_solve/solutionX/{i}/pytorch_{j}")
computed_energy = verify_solution(Q, solution)

with open(f"solution/{i}/pytorch_{j}", "w") as f:
    f.write(f"Pytorch Energy: {computed_energy}\n")

