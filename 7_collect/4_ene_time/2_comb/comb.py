from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("time_file", type=str)
parser.add_argument("ene_file", type=str)
parser.add_argument("output_file", type=str)
args = parser.parse_args()

time_file = Path(args.time_file)
ene_file = Path(args.ene_file)
output_file = Path(args.output_file)

thresholds = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]

data_order = [
    "1_data_1000",
    "2_data_6000",
    "3_data_25000",
    "4_data_45000",
]


def read_block_file(filename):
    data_dict = {}
    current_data = None
    current_thresholds = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line == "":
                continue

            parts = line.split()

            if len(parts) == 1 and "_data_" in parts[0]:
                current_data = parts[0]
                data_dict[current_data] = {}
                continue

            if parts[0] == "method":
                current_thresholds = parts[1:]
                continue

            if current_data is not None:
                method = parts[0]
                values = parts[1:]

                data_dict[current_data][method] = {
                    th: val for th, val in zip(current_thresholds, values)
                }

    return data_dict


time_data = read_block_file(time_file)
ene_data = read_block_file(ene_file)

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as out:
    for data in data_order:
        if data not in time_data and data not in ene_data:
            continue

        out.write(f"{data}\n")
        out.write(f"{'method':<22}")

        for th in thresholds:
            out.write(f"{th:>16}")

        out.write("\n")

        methods = sorted(
            set(time_data.get(data, {}).keys()) | set(ene_data.get(data, {}).keys())
        )

        for method in methods:
            if method in time_data.get(data, {}):
                out.write(f"{method + '_time':<22}")

                for th in thresholds:
                    value = time_data[data][method].get(th, "")
                    out.write(f"{value:>16}")

                out.write("\n")

            if method in ene_data.get(data, {}):
                out.write(f"{method + '_energy':<22}")

                for th in thresholds:
                    value = ene_data[data][method].get(th, "")
                    out.write(f"{value:>16}")

                out.write("\n")

        out.write("\n")

