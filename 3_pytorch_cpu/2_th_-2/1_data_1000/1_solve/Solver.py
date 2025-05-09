import sys
import time
import numpy as np
import torch
import warnings
from torch import optim

warnings.filterwarnings("ignore", message="Attempting to run cuBLAS, but there was no current CUDA context!")

Num = 1000
dev = 'cpu'
slop = 0.5
threshold = 1e-2
learning_rate = 1e-2
Num_Step = 1000000

def load_Q(filename):
    Q_np = np.load(filename)
    return torch.tensor(Q_np, dtype=torch.float32, device=dev)

def Train_X(Opt, Q, X):
    Opt.zero_grad()
    X2 = torch.sigmoid(slop * (X - 0.5)).reshape(1, -1)
    loss = (X2 @ Q @ X2.T).squeeze()
    loss.backward()
    Opt.step()
    X.data.copy_(torch.clamp(X.data, -5, 5))
    return X, loss

def solve_with_pytorch(Q):
    start_time = time.time()
    X = torch.randn((Q.shape[0]), device=dev, dtype=torch.float, requires_grad=True)
    Opt = optim.Adam([X], lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Opt, mode='min', factor=0.5, patience=500, min_lr=1e-6)

    window_size = min(max(500, Num_Step // 100), 2000)
    loss_history = []
    min_loss = float("inf")
    patience = max(int(scheduler.patience * 1.5), 800)
    patience_counter = 0

    for step in range(1, Num_Step + 1):
        X, loss = Train_X(Opt, Q, X)
        loss_value = loss.item()
        loss_history.append(loss_value)

        if step >= window_size:
            recent_losses = loss_history[-window_size:]
            loss_avg = sum(recent_losses) / len(recent_losses)
            loss_change = abs(loss_avg - min(recent_losses)) / max(abs(min(recent_losses)), 1e-8)

            if loss_change < threshold:
                scheduler.step(loss_value)

            if loss_change < threshold or patience_counter > patience:
                break

        if loss_value < min_loss:
            min_loss = loss_value
            patience_counter = 0
        else:
            patience_counter += 1

    elapsed_time = time.time() - start_time
    pytorch_solution = (torch.sigmoid(slop * (X - 0.5)) > 0.5).float()
    solution_2d = pytorch_solution.unsqueeze(0)
    upper_Q = torch.triu(Q)
    pytorch_energy = (solution_2d @ upper_Q @ solution_2d.t()).item()

    return pytorch_solution, pytorch_energy, elapsed_time, step

if __name__ == "__main__":
    i = sys.argv[1]
    j = sys.argv[2]
    Q_filename = f"../../../../1_gen_coe/1_data_{Num}/data_{i}.npy"
    result_filename = f"solution/{i}/pytorch_{j}"
    solution_filename = f"solutionX/{i}/pytorch_{j}"

    Q = load_Q(Q_filename)
    pytorch_solution, pytorch_energy, pytorch_time, step_num = solve_with_pytorch(Q)

    with open(result_filename, "w") as f:
        f.write(f"PyTorch Energy: {pytorch_energy}\n")
        f.write(f"PyTorch Time: {pytorch_time:.6f} sec\n")
        f.write(f"PyTorch Steps: {step_num}\n") 

    pytorch_solution = pytorch_solution.to(torch.int)
    with open(solution_filename, "w") as f:
        f.write("PyTorch Solution:\n")
        f.write(" ".join(map(str, pytorch_solution.tolist())) + "\n")


