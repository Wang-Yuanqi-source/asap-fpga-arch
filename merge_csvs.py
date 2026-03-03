import csv
import glob
import os
from typing import List


def merge_csv_files(input_pattern: str, output_path: str, benchmark_column: str = "benchmark") -> None:
    files: List[str] = sorted(glob.glob(input_pattern))
    if not files:
        raise SystemExit(f"No CSV files matched pattern: {input_pattern}")

    first_header_written = False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)

        for path in files:
            benchmark_name = os.path.splitext(os.path.basename(path))[0]

            with open(path, newline="", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)

                try:
                    header = next(reader)
                except StopIteration:
                    # skip completely empty files
                    continue

                if not first_header_written:
                    writer.writerow(header + [benchmark_column])
                    first_header_written = True

                for row in reader:
                    writer.writerow(row + [benchmark_name])


if __name__ == "__main__":
    merge_csv_files(input_pattern=os.path.join("data", "*.csv"), output_path=os.path.join("data", "merged.csv"))

