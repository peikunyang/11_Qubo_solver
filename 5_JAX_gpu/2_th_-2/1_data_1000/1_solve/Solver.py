import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

Num = 1000
slop = 0.5
threshold = 1e-2
learning_rate = 1e-2
Num_Step = 1000000

optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-5)

def sigmoid_scaled(x, slope):
    return jax.nn.sigmoid(slope * (x - 0.5))

def update(params, opt_state, Q, slope):
    def loss_fn_local(x):
        x2 = sigmoid_scaled(x, slope).reshape(1, -1)
        return (x2 @ Q @ x2.T).squeeze()

    loss, grads = jax.value_and_grad(loss_fn_local)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    params = jnp.clip(params, -5.0, 5.0)
    return params, opt_state, loss

def solve_with_jax_debug(Q, n, loss_log_filename):
    start_time = time.time()
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, shape=(n,))

    opt_state = optimizer.init(x)
    loss_history = []
    min_loss = float("inf")
    patience_counter = 0

    with open(loss_log_filename, "w") as logf:
        logf.write(f"{'Step':<8}{'Loss':>12}\n")
        logf.write("-" * 20 + "\n")

        for step in range(1, Num_Step + 1):
            x, opt_state, loss = update(x, opt_state, Q, slop)
            loss_val = float(loss)
            logf.write(f"{step:<8}{loss_val:>12.6f}\n")

            loss_history.append(loss_val)

            window_size = min(max(50, Num_Step // 10), 200)
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

    elapsed_time = time.time() - start_time
    final_sigmoid = sigmoid_scaled(x, slop)
    binary_solution = (final_sigmoid > 0.5).astype(jnp.float32)
    upper_Q = jnp.triu(Q)
    energy = float((binary_solution @ upper_Q @ binary_solution.T))

    return binary_solution.astype(int), energy, elapsed_time, step

if __name__ == "__main__":
    i = sys.argv[1]
    j = sys.argv[2]
    Q_filename = f"../../../../1_gen_coe/1_data_{Num}/data_{i}.npy"
    result_filename = f"solution/{i}/jax_{j}"
    solution_filename = f"solutionX/{i}/jax_{j}"
    loss_log_filename = f"loss/{i}/jax_{j}_loss.txt"

    os.makedirs(f"solution/{i}", exist_ok=True)
    os.makedirs(f"solutionX/{i}", exist_ok=True)

    Q_np = np.load(Q_filename)
    n = Q_np.shape[0]
    Q = jnp.array(Q_np)

    jax_solution, jax_energy, jax_time, step_num = solve_with_jax_debug(Q, n, loss_log_filename)

    with open(result_filename, "w") as f:
        f.write(f"JAX Energy: {jax_energy}\n")
        f.write(f"JAX Time: {jax_time:.6f} sec\n")
        f.write(f"PyTorch Steps: {step_num}\n")  # 新增這行

    with open(solution_filename, "w") as f:
        f.write("JAX Solution:\n")
        f.write(" ".join(map(str, jax_solution.tolist())) + "\n")

