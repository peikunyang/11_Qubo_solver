import sys
import re
import numpy as np
from pathlib import Path

solve_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../1_solve")

datas = ["1_data_1000", "2_data_6000", "3_data_25000"]

num_threshold = 6
num_data_id = 5
num_seed = 5

solver_name = "pytorch"

number_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def read_solver_results(filename):
    energy = None
    time_used = None
    step = None

    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        if "Energy" in line:
            match = re.search(number_pattern, line)
            if match is not None:
                energy = float(match.group())

        elif "Time" in line:
            match = re.search(number_pattern, line)
            if match is not None:
                time_used = float(match.group())

        elif "Steps" in line:
            match = re.search(number_pattern, line)
            if match is not None:
                step = int(float(match.group()))

    if energy is None:
        raise ValueError(f"Energy not found: {filename}")

    if time_used is None:
        raise ValueError(f"Time not found: {filename}")

    if step is None:
        raise ValueError(f"Steps not found: {filename}")

    return energy, time_used, step


Ene = []
Time = []
Step = []

for i in range(1, num_threshold + 1):
    threshold_folder = f"thr_1e-{i}"

    for data in datas:
        for j in range(1, num_data_id + 1):
            data_id_folder = f"data_{j}"

            for k in range(1, num_seed + 1):
                filename = (
                    solve_root
                    / "solution"
                    / data
                    / threshold_folder
                    / data_id_folder
                    / f"{solver_name}_{k}"
                )

                if not filename.is_file():
                    raise FileNotFoundError(f"File not found: {filename}")

                energy, time_used, step = read_solver_results(filename)

                Ene.append(energy)
                Time.append(time_used)
                Step.append(step)

len_data = len(datas)

Ene_array = np.array(Ene).reshape(num_threshold, len_data, num_data_id, num_seed)
Time_array = np.array(Time).reshape(num_threshold, len_data, num_data_id, num_seed)
Step_array = np.array(Step).reshape(num_threshold, len_data, num_data_id, num_seed)

Enen = Ene_array.min(axis=3)
Timen = Time_array.mean(axis=3).mean(axis=2)
Stepn = Step_array.mean(axis=3).mean(axis=2)

with open("3_PyTorch_ene", "w") as file:
    for i in range(len_data):
        file.write("%s\n" % datas[i])

        for m in range(num_threshold):
            for j in range(num_data_id):
                file.write("%12.5e " % Enen[m][i][j])
            file.write("\n")

        file.write("\n")

    file.write("\n")

with open("3_PyTorch_time", "w") as file:
    for i in range(len_data):
        file.write("%s\n" % datas[i])

        for m in range(num_threshold):
            file.write("%8.3f\n" % Timen[m][i])

        file.write("\n")

    file.write("\n")

with open("3_PyTorch_step", "w") as file:
    for i in range(len_data):
        file.write("%s\n" % datas[i])

        for m in range(num_threshold):
            file.write("%8.2f\n" % Stepn[m][i])

        file.write("\n")

    file.write("\n")

print("Done")
print("Output files:")
print("4_PyTorch_ene")
print("4_PyTorch_time")
print("4_PyTorch_step")

