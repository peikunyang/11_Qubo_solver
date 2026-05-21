import sys
import re
import numpy as np
from pathlib import Path

solve_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../1_solve")

datas = ["1_data_1000", "2_data_6000"]

num_data_id = 5
num_seed = 5

number_pattern = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def read_pyqubo_results(filename):
    energy = None
    time_used = None

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

    if energy is None:
        raise ValueError(f"Energy not found: {filename}")

    if time_used is None:
        raise ValueError(f"Time not found: {filename}")

    return energy, time_used


Ene = []
Time = []

for data in datas:
    for j in range(1, num_data_id + 1):
        data_id_folder = f"data_{j}"

        for k in range(1, num_seed + 1):
            filename = (
                solve_root
                / "solution"
                / data
                / data_id_folder
                / f"pyqubo_{k}"
            )

            if not filename.is_file():
                raise FileNotFoundError(f"File not found: {filename}")

            energy, time_used = read_pyqubo_results(filename)

            Ene.append(energy)
            Time.append(time_used)


len_data = len(datas)

Ene_array = np.array(Ene).reshape(len_data, num_data_id, num_seed)
Time_array = np.array(Time).reshape(len_data, num_data_id, num_seed)

Enen = Ene_array.min(axis=2)
Timen = Time_array.mean(axis=2).mean(axis=1)


with open("2_neal_ene", "w") as file:
    for i in range(len_data):
        file.write("%s\n" % datas[i])

        for j in range(num_data_id):
            file.write("%12.5e " % Enen[i][j])

        file.write("\n\n")


with open("2_neal_time", "w") as file:
    for i in range(len_data):
        file.write("%s\n" % datas[i])
        file.write("%8.3f\n" % Timen[i])
        file.write("\n")


print("Done")
print("Output files:")
print("2_neal_ene")
print("2_neal_time")

