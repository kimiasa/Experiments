import argparse
import triton.profiler.viewer as protonviewer
from triton.profiler.hook import COMPUTE_METADATA_SCOPE_NAME
import pandas as pd

import os

TOTAL_REPETITIONS = 25

def parse_benchmark(metrics, filename):
    with open(filename, "r") as f:
        gf, raw_metrics = protonviewer.get_raw_metrics(f)
        assert len(raw_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        metrics = protonviewer.derive_metrics(gf, metrics, raw_metrics)

        def filter_autotune(x):
            parents = x.node.path()
            for p in parents:
                if p.frame['name'] == 'forward':
                    return True
            return False
        gf.dataframe[f"avgTime (inc)"] = (gf.dataframe['time/ms (inc)'] / TOTAL_REPETITIONS)

        gf = gf.filter(lambda x: filter_autotune(x), squash=True)
        filtered_df = gf.dataframe[gf.dataframe['name'] == 'forward']
        if filtered_df.shape[0] != 1:
            raise RuntimeError(f"{filename} has multiple forward frames")
        
        filtered_df['experiment'] = filename.split('/')[-1].replace('.hatchet', '')
        filtered_df.set_index('experiment', inplace=True)
        return filtered_df


def main(directory, csv_output_path):
    if not csv_output_path:
        csv_output_path = "benchmarks.csv"
    if not os.path.isdir(directory):
        print(f'{directory} is not a valid directory, please enter a valid path')
        return

    dataframes = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and file_path.endswith('.hatchet'):
            print(f"Processing .hatchet file: {filename}")
            dataframes.append(parse_benchmark(['time/ms', 'count'], file_path))
            df = pd.concat(dataframes)
            df.to_csv(csv_output_path)
            print(f"Saved CSV to {csv_output_path}")

        else:
            print(f"Skipping non-.hatchet file: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files in a specified directory.")
    parser.add_argument("--directory", help="Path to the directory to process")
    parser.add_argument("--csv_output_path", help="Path to the csv outfile file")
    args = parser.parse_args()
    directory = args.directory
    csv_output_path = args.csv_output_path
    main(directory, csv_output_path)