## Release Notes

This repository is a fork of [https://github.com/HazyResearch/fly](https://github.com/HazyResearch/fly), specifically adapted to benchmark using SS1, Monarch, and low-rank matrices within the GPT2 architecture, with the standard model as baseline. This repository includes modifications to compare performance across these configurations, aligned with the experiments discussed in our paper.

### Inference benchmarking for SSLinear
All experiment configs for SSLinear are specified in `configs/experiments/ssl_bench_mlp`. 
These experiments run on GPT2 for 25 iterations, and generate `.hatchet` files using proton
to the `./proton_benchmarks` directory. They run in batch sizes of `8, 16, 32`.

0. First, be sure to init all the submodules. To modify any parameters shared by multiple runs
(e.g. number of devices, base model, etc.), you can modify `configs/experiments/ssl_bench_mlp/base_gpt2.yaml`

1. To run the experiment script, run the following:
```sh
./run_benchmarks.sh configs/experiment/ssl_bench_mlp/
```
Optionally, you can pass in the `--ignore-failure` flag if you want the benchmarking script to ignore failures (e.g. OOM errors)
and continue running other configurations.

2. To generate a unified CSV with only the timings for the `forward` step, run
```sh
python generate_benchmark_csv.py --directory proton_benchmarks/ --csv_output_path benchmarkfile.csv
```
Which will generate a csv file with all of the experimental configs, and in the `avgTime (inc)` column,
show the average time a single inference batch took.




