import sys
import re
import csv
import torch
import numpy as np
from pathlib import Path

solve_root = Path(sys.argv[1])
data_root = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("../../../../1_gen_coe")

solutionX_root = solve_root / "solutionX"
pyqubo_result_root = solve_root / "solution"

output_root = Path("solution")
summary_file = Path("verify_pyqubo_energy_summary.csv")

abs_tol = 1e-4
rel_tol = 1e-6

number_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def sort_key(path):
    nums = re.findall(r"\d+", path.name)
    if nums:
        return [int(x) for x in nums]
    return [path.name]


def load_Q(filename):
    Q_np = np.load(filename).astype(np.float32, copy=False)
    Q = torch.tensor(Q_np, dtype=torch.float32, device="cpu")
    Q = torch.triu(Q)
    return Q


def load_solution(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"Invalid solution file: {filename}")

    solution = list(map(int, lines[1].strip().split()))
    return torch.tensor(solution, dtype=torch.int64)


def read_pyqubo_energy(filename):
    if not filename.is_file():
        return None

    with open(filename, "r") as f:
        for line in f:
            if "PyQUBO Energy" in line or "Energy" in line:
                match = re.search(number_pattern, line)
                if match is not None:
                    return float(match.group())

    return None


def verify_solution(Q, solution):
    x = solution.to(Q.dtype)
    return float(x @ Q @ x)


def is_match(a, b):
    if a is None or b is None:
        return False

    diff = abs(a - b)
    tolerance = abs_tol + rel_tol * abs(b)

    return diff <= tolerance


if not solutionX_root.is_dir():
    raise NotADirectoryError(f"solutionX folder not found: {solutionX_root}")

if not pyqubo_result_root.is_dir():
    raise NotADirectoryError(f"solution folder not found: {pyqubo_result_root}")


rows = []

for data_folder_path in sorted(solutionX_root.iterdir(), key=sort_key):
    if not data_folder_path.is_dir():
        continue

    data_folder = data_folder_path.name

    for data_id_path in sorted(data_folder_path.iterdir(), key=sort_key):
        if not data_id_path.is_dir():
            continue

        data_id_folder = data_id_path.name

        Q_filename = data_root / data_folder / f"{data_id_folder}.npy"

        if not Q_filename.is_file():
            rows.append([
                data_folder,
                data_id_folder,
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                "Q_FILE_NOT_FOUND"
            ])
            print(data_folder, data_id_folder, "Q_FILE_NOT_FOUND")
            continue

        Q = load_Q(Q_filename)

        for solution_file in sorted(data_id_path.iterdir(), key=sort_key):
            if not solution_file.is_file():
                continue

            if not solution_file.name.startswith("pyqubo_"):
                continue

            seed = solution_file.name.replace("pyqubo_", "", 1)

            pyqubo_result_file = (
                pyqubo_result_root
                / data_folder
                / data_id_folder
                / f"pyqubo_{seed}"
            )

            output_folder = (
                output_root
                / data_folder
                / data_id_folder
            )

            output_folder.mkdir(parents=True, exist_ok=True)

            output_file = output_folder / f"pytorch_verify_pyqubo_{seed}"

            try:
                solution = load_solution(solution_file)

                if Q.shape[0] != solution.shape[0]:
                    raise ValueError(
                        f"Size mismatch: Q size {Q.shape[0]}, solution size {solution.shape[0]}"
                    )

                verified_energy = verify_solution(Q, solution)
                reported_energy = read_pyqubo_energy(pyqubo_result_file)

                if reported_energy is None:
                    diff = "NA"
                    status = "PYQUBO_ENERGY_NOT_FOUND"
                else:
                    diff_value = verified_energy - reported_energy
                    diff = diff_value

                    if is_match(verified_energy, reported_energy):
                        status = "MATCH"
                    else:
                        status = "DIFF"

                with open(output_file, "w") as f:
                    f.write(f"PyTorch Verified Energy: {verified_energy}\n")

                    if reported_energy is not None:
                        f.write(f"PyQUBO Reported Energy: {reported_energy}\n")
                        f.write(f"Energy Difference: {verified_energy - reported_energy}\n")
                        f.write(f"Status: {status}\n")
                    else:
                        f.write("PyQUBO Reported Energy: NA\n")
                        f.write("Energy Difference: NA\n")
                        f.write(f"Status: {status}\n")

                rows.append([
                    data_folder,
                    data_id_folder,
                    seed,
                    verified_energy,
                    reported_energy if reported_energy is not None else "NA",
                    diff,
                    str(pyqubo_result_file),
                    status
                ])

                print(data_folder, data_id_folder, seed, status)

            except Exception as e:
                rows.append([
                    data_folder,
                    data_id_folder,
                    seed,
                    "NA",
                    "NA",
                    "NA",
                    str(pyqubo_result_file),
                    f"ERROR: {e}"
                ])

                print(data_folder, data_id_folder, seed, "ERROR", e)


with open(summary_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "data_folder",
        "data_id",
        "seed",
        "pytorch_verified_energy",
        "pyqubo_reported_energy",
        "energy_difference",
        "pyqubo_result_file",
        "status"
    ])
    writer.writerows(rows)

print(f"Done. Summary written to {summary_file}")

