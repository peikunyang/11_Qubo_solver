from pathlib import Path
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str)
parser.add_argument("output_file", type=str)
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_file = Path(args.output_file)

files = [
    "1_data_1000",
    "2_data_6000",
    "3_data_25000",
    "4_data_45000",
]

thresholds = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]

if not input_dir.is_dir():
    raise NotADirectoryError(f"input directory not found: {input_dir}")

output_file.parent.mkdir(parents=True, exist_ok=True)


def read_file(filename):
    blocks = OrderedDict()
    current_data = None
    current_thresholds = None

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line == "":
                continue

            if line.startswith("data_"):
                current_data = line
                blocks[current_data] = OrderedDict()
                current_thresholds = None
                continue

            parts = line.split()

            if parts[0] == "method":
                current_thresholds = parts[1:]
                continue

            if current_data is not None and current_thresholds is not None:
                method = parts[0]
                values = [float(x) for x in parts[1:]]
                blocks[current_data][method] = dict(zip(current_thresholds, values))

    return blocks


def average_blocks(blocks):
    sums = OrderedDict()
    counts = OrderedDict()

    for data_name, table in blocks.items():
        for method, row in table.items():
            if method not in sums:
                sums[method] = OrderedDict()
                counts[method] = OrderedDict()

            for threshold, value in row.items():
                if threshold not in sums[method]:
                    sums[method][threshold] = 0.0
                    counts[method][threshold] = 0

                sums[method][threshold] += value
                counts[method][threshold] += 1

    averages = OrderedDict()

    for method in sums:
        averages[method] = OrderedDict()
        for threshold in thresholds:
            if threshold in sums[method]:
                averages[method][threshold] = sums[method][threshold] / counts[method][threshold]

    return averages


with open(output_file, "w", encoding="utf-8") as out:
    for file_name in files:
        filename = input_dir / file_name

        if not filename.is_file():
            print(f"skip missing file: {filename}")
            continue

        blocks = read_file(filename)
        averages = average_blocks(blocks)

        out.write(f"{file_name}\n")
        out.write(f"{'method':<14}")

        for threshold in thresholds:
            out.write(f"{threshold:>16}")

        out.write("\n")

        for method, row in averages.items():
            out.write(f"{method:<14}")

            for threshold in thresholds:
                value = row.get(threshold)

                if value is None:
                    out.write(f"{'nan':>16}")
                else:
                    out.write(f"{value:>16.5e}")

            out.write("\n")

        out.write("\n")

print(f"saved to: {output_file}")

