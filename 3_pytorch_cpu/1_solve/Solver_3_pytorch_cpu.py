import sys
import os
import time
import numpy as np
import torch
from torch import optim

dev = "cpu"
slop = 0.5
learning_rate = 1e-2
Num_Step = 1000000

data_root = sys.argv[1]
data_folder = sys.argv[2]
threshold_name = sys.argv[3]
data_id = sys.argv[4]
seed = int(sys.argv[5])

threshold = float(threshold_name)

def load_Q(filename):
    Q_np = np.load(filename)
    Q = torch.tensor(Q_np, dtype=torch.float32, device=dev)
    Q.triu_()
    return Q

def Train_X(Opt, Q, X):
    Opt.zero_grad()
    X2 = torch.sigmoid(slop * (X - 0.5)).reshape(1, -1)
    loss = (X2 @ Q @ X2.T).squeeze()
    loss.backward()
    Opt.step()
    X.data.copy_(torch.clamp(X.data, -5, 5))
    return X, loss

def solve_with_pytorch(Q, threshold, seed):
    torch.manual_seed(seed)

    X = torch.randn((Q.shape[0]), device=dev, dtype=torch.float, requires_grad=True)
    Opt = optim.Adam([X], lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        Opt,
        mode="min",
        factor=0.5,
        patience=500,
        min_lr=1e-6
    )

    window_size = min(max(500, Num_Step // 100), 2000)
    loss_history = []
    min_loss = float("inf")
    patience = max(int(scheduler.patience * 1.5), 800)
    patience_counter = 0

    start_time = time.time()

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
    pytorch_energy = (solution_2d @ Q @ solution_2d.t()).item()

    return pytorch_solution, pytorch_energy, elapsed_time, step

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

result_filename = os.path.join(result_folder, f"pytorch_{seed}")
solution_filename = os.path.join(solution_folder, f"pytorch_{seed}")

Q = load_Q(Q_filename)

pytorch_solution, pytorch_energy, pytorch_time, step_num = solve_with_pytorch(
    Q,
    threshold,
    seed
)

with open(result_filename, "w") as f:
    f.write(f"PyTorch Energy: {pytorch_energy}\n")
    f.write(f"PyTorch Time: {pytorch_time:.6f} sec\n")
    f.write(f"PyTorch Steps: {step_num}\n")

pytorch_solution = pytorch_solution.to(torch.int)

with open(solution_filename, "w") as f:
    f.write("PyTorch Solution:\n")
    f.write(" ".join(map(str, pytorch_solution.tolist())) + "\n")

