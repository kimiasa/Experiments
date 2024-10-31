## Release Notes

This repository is a fork of [https://github.com/HazyResearch/fly](https://github.com/HazyResearch/fly), specifically adapted to benchmark using SS1, Monarch, and low-rank matrices within the GPT2 architecture, with the standard model as baseline. This repository includes modifications to compare performance across these configurations, aligned with the experiments discussed in our paper.

Credits for the original codebase go to the initial contributors, with all new modifications for our benchmarking and analysis purposes.

## Setup 
```sh
pip install -r requirements.txt
```
### Training
Mixed-precision training (using FP16 tensors) is supported on A100 GPUs and newer models.
### SS1
```sh
python run.py experiment=wt103/gpt2-ssl
```
### Monarch
```sh
python run.py experiment=wt103/monarch
```
### Low-Rank
```sh
python run.py experiment=wt103/lowrank
```





