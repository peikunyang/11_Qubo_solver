import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

slop = 0.5
learning_rate = 1e-2
Num_Step = 1000000

data_root = sys.argv[1]
data_folder = sys.argv[2]
threshold_name = sys.argv[3]
data_id = sys.argv[4]
seed = int(sys.argv[5])

threshold = float(threshold_name)

gpu_device = jax.devices("gpu")[0]

optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)

def sigmoid_scaled(x, slope):
    return jax.nn.sigmoid(slope * (x - 0.5))

def load_Q(filename):
    Q_np = np.load(filename)
    Q = jnp.array(Q_np, dtype=jnp.float32)
    Q = jnp.triu(Q)
    Q = jax.device_put(Q, gpu_device)
    return Q

@jax.jit
def update(params, opt_state, Q, slope):
    def loss_fn_local(x):
        x2 = sigmoid_scaled(x, slope).reshape(1, -1)
        return (x2 @ Q @ x2.T).squeeze()

    loss, grads = jax.value_and_grad(loss_fn_local)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    params = jnp.clip(params, -5.0, 5.0)

    return params, opt_state, loss

def solve_with_jax(Q, n, threshold, seed):
    key = jax.random.PRNGKey(seed)
    x0 = jax.random.normal(key, shape=(n,))
    x0 = jax.device_put(x0, gpu_device)

    opt_state0 = optimizer.init(x0)

    warm_x, warm_opt_state, warm_loss = update(x0, opt_state0, Q, slop)
    warm_x.block_until_ready()

    x = x0
    opt_state = opt_state0

    loss_history = []
    min_loss = float("inf")
    patience_counter = 0

    window_size = min(max(50, Num_Step // 10), 200)

    start_time = time.time()

    for step in range(1, Num_Step + 1):
        x, opt_state, loss = update(x, opt_state, Q, slop)
        loss_val = float(loss)
        loss_history.append(loss_val)

        if step >= window_size:
            recent = loss_history[-window_size:]
            loss_avg = sum(recent) / len(recent)
            loss_change = abs(loss_avg - min(recent)) / max(abs(min(recent)), 1e-8)

            if loss_change < threshold or patience_counter > 200:
                break

        if loss_val < min_loss:
            min_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

    x.block_until_ready()
    elapsed_time = time.time() - start_time

    final_sigmoid = sigmoid_scaled(x, slop)
    binary_solution = (final_sigmoid > 0.5).astype(jnp.int32)
    binary_solution.block_until_ready()

    return binary_solution, elapsed_time, step, loss_history

def compute_energy_cpu(Q_filename, solution):
    Q_cpu = np.load(Q_filename).astype(np.float64, copy=False)
    upper_Q_cpu = np.triu(Q_cpu)

    solution_cpu = np.asarray(jax.device_get(solution), dtype=np.float64)

    energy = float(solution_cpu @ upper_Q_cpu @ solution_cpu.T)

    return energy

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

loss_folder = os.path.join(
    "loss",
    data_folder,
    f"thr_{threshold_name}",
    f"data_{data_id}"
)

os.makedirs(result_folder, exist_ok=True)
os.makedirs(solution_folder, exist_ok=True)
os.makedirs(loss_folder, exist_ok=True)

result_filename = os.path.join(result_folder, f"jax_{seed}")
solution_filename = os.path.join(solution_folder, f"jax_{seed}")
loss_log_filename = os.path.join(loss_folder, f"jax_{seed}_loss.txt")

Q = load_Q(Q_filename)
n = Q.shape[0]

jax_solution, jax_time, step_num, loss_history = solve_with_jax(
    Q,
    n,
    threshold,
    seed
)

jax_energy = compute_energy_cpu(Q_filename, jax_solution)

jax_solution_cpu = np.asarray(jax.device_get(jax_solution), dtype=np.int32)

with open(result_filename, "w") as f:
    f.write(f"JAX Energy: {jax_energy}\n")
    f.write(f"JAX Time: {jax_time:.6f} sec\n")
    f.write(f"JAX Steps: {step_num}\n")

with open(solution_filename, "w") as f:
    f.write("JAX Solution:\n")
    f.write(" ".join(map(str, jax_solution_cpu.tolist())) + "\n")

with open(loss_log_filename, "w") as f:
    f.write(f"{'Step':<8}{'Loss':>12}\n")
    f.write("-" * 20 + "\n")
    for idx, loss_val in enumerate(loss_history, start=1):
        f.write(f"{idx:<8}{loss_val:>12.6f}\n")

