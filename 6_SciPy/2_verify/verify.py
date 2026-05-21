import sys
import re
import csv
import torch
import numpy as np
from pathlib import Path

solve_root = Path(sys.argv[1])
data_root = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("../../../../1_gen_coe")

solutionX_root = solve_root / "solutionX"
scipy_solution_root = solve_root / "solution"

output_root = Path("solution")
summary_file = Path("verify_scipy_energy_summary.csv")

abs_tol = 1e-6
rel_tol = 1e-10

number_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def sort_key(path):
    nums = re.findall(r"\d+", path.name)
    if nums:
        return [int(x) for x in nums]
    return [path.name]


def load_Q(filename):
    Q_np = np.load(filename).astype(np.float64, copy=False)
    return torch.tensor(Q_np, dtype=torch.float64)


def load_solution(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"Invalid solution file: {filename}")

    solution = list(map(int, lines[1].strip().split()))
    return torch.tensor(solution, dtype=torch.int64)


def read_scipy_energy(filename):
    if not filename.is_file():
        return None

    with open(filename, "r") as f:
        for line in f:
            if "SciPy Energy" in line or "Energy" in line:
                match = re.search(number_pattern, line)
                if match is not None:
                    return float(match.group())

    return None


def verify_solution(Q, solution):
    upper_Q = torch.triu(Q)
    x = solution.to(Q.dtype)
    return float(x @ upper_Q @ x)


def is_match(a, b):
    if a is None or b is None:
        return False

    diff = abs(a - b)
    tolerance = abs_tol + rel_tol * abs(b)

    return diff <= tolerance


if not solutionX_root.is_dir():
    raise NotADirectoryError(f"solutionX folder not found: {solutionX_root}")

if not scipy_solution_root.is_dir():
    raise NotADirectoryError(f"solution folder not found: {scipy_solution_root}")

rows = []

for data_folder_path in sorted(solutionX_root.iterdir(), key=sort_key):
    if not data_folder_path.is_dir():
        continue

    data_folder = data_folder_path.name

    for threshold_path in sorted(data_folder_path.iterdir(), key=sort_key):
        if not threshold_path.is_dir():
            continue

        threshold_folder = threshold_path.name

        if not threshold_folder.startswith("thr_"):
            continue

        threshold_name = threshold_folder.replace("thr_", "", 1)

        for data_id_path in sorted(threshold_path.iterdir(), key=sort_key):
            if not data_id_path.is_dir():
                continue

            data_id_folder = data_id_path.name

            Q_filename = data_root / data_folder / f"{data_id_folder}.npy"

            if not Q_filename.is_file():
                rows.append([
                    data_folder,
                    threshold_name,
                    data_id_folder,
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "Q_FILE_NOT_FOUND"
                ])
                print(data_folder, threshold_name, data_id_folder, "Q_FILE_NOT_FOUND")
                continue

            Q = load_Q(Q_filename)

            for solution_file in sorted(data_id_path.iterdir(), key=sort_key):
                if not solution_file.is_file():
                    continue

                if not solution_file.name.startswith("scipy_"):
                    continue

                seed = solution_file.name.replace("scipy_", "", 1)

                scipy_result_file = (
                    scipy_solution_root
                    / data_folder
                    / threshold_folder
                    / data_id_folder
                    / f"scipy_{seed}"
                )

                output_folder = (
                    output_root
                    / data_folder
                    / threshold_folder
                    / data_id_folder
                )

                output_folder.mkdir(parents=True, exist_ok=True)

                output_file = output_folder / f"pytorch_{seed}"

                try:
                    solution = load_solution(solution_file)

                    if Q.shape[0] != solution.shape[0]:
                        raise ValueError(
                            f"Size mismatch: Q size {Q.shape[0]}, solution size {solution.shape[0]}"
                        )

                    pytorch_energy = verify_solution(Q, solution)
                    scipy_energy = read_scipy_energy(scipy_result_file)

                    if scipy_energy is None:
                        diff = "NA"
                        status = "SCIPY_ENERGY_NOT_FOUND"
                    else:
                        diff_value = pytorch_energy - scipy_energy
                        diff = diff_value

                        if is_match(pytorch_energy, scipy_energy):
                            status = "MATCH"
                        else:
                            status = "DIFF"

                    with open(output_file, "w") as f:
                        f.write(f"Pytorch Verified Energy: {pytorch_energy}\n")

                        if scipy_energy is not None:
                            f.write(f"SciPy Reported Energy: {scipy_energy}\n")
                            f.write(f"Energy Difference: {pytorch_energy - scipy_energy}\n")
                            f.write(f"Status: {status}\n")
                        else:
                            f.write("SciPy Reported Energy: NA\n")
                            f.write("Energy Difference: NA\n")
                            f.write(f"Status: {status}\n")

                    rows.append([
                        data_folder,
                        threshold_name,
                        data_id_folder,
                        seed,
                        pytorch_energy,
                        scipy_energy if scipy_energy is not None else "NA",
                        diff,
                        str(scipy_result_file),
                        status
                    ])

                    print(
                        data_folder,
                        threshold_name,
                        data_id_folder,
                        seed,
                        status
                    )

                except Exception as e:
                    rows.append([
                        data_folder,
                        threshold_name,
                        data_id_folder,
                        seed,
                        "NA",
                        "NA",
                        "NA",
                        str(scipy_result_file),
                        f"ERROR: {e}"
                    ])

                    print(
                        data_folder,
                        threshold_name,
                        data_id_folder,
                        seed,
                        "ERROR",
                        e
                    )

with open(summary_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "data_folder",
        "threshold",
        "data_id",
        "seed",
        "pytorch_verified_energy",
        "scipy_reported_energy",
        "energy_difference",
        "scipy_result_file",
        "status"
    ])
    writer.writerows(rows)

print(f"Done. Summary written to {summary_file}")

