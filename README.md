## Release Notes

This repository is a fork of [https://github.com/HazyResearch/fly](https://github.com/HazyResearch/fly), specifically adapted to benchmark the use of SS1, Monarch, and low-rank matrices within the GPT2 architecture, with the standard model as baseline. This repository includes modifications to compare performance across these configurations, aligned with the experiments discussed in our paper.

Credits for the original codebase go to the initial contributors, with all new modifications for our benchmarking and analysis purposes.

## Requirements

Python 3.8+, Pytorch 1.9+, torchvision, torchtext, pytorch-fast-transformers, munch, einops, timm, hydra-core, hydra-colorlog, python-dotenv, rich, pytorch-lightning, lightning-bolts, triton.

There is a Dockerfile that lists all the required packages.

## Training SS1-GPT2 

In order to train G

### T2T-ViT inference on ImageNet
To run the T2T-ViT inference on ImageNet experiment:
1. Download the pretrained weights from the [T2T-ViT repo][https://github.com/yitu-opensource/T2T-ViT/releases]:
```sh
mkdir -p checkpoints/t2tvit
cd checkpoints/t2tvit
wget https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.7_T2T_ViTt_14.pth.tar
```
2. Convert the weights to the format compatible with our implementation of
   T2T-ViT:
```sh
# cd to scatterbrain path
python scripts/convert_checkpoint_t2t_vit.py checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar
```
3. Download the ImageNet dataset (just the validation set will suffice).
Below, `/path/to/imagenet` refers to the directory that contains the `train` and `val` directories.
4. Run the inference experiments:
```sh
python run.py experiment=imagenet-t2tvit-eval.yaml model/t2tattn_cfg=full datamodule.data_dir=/path/to/imagenet/ eval.ckpt=checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar  # 81.7% acc
python run.py experiment=imagenet-t2tvit-eval.yaml model/t2tattn_cfg=local datamodule.data_dir=/path/to/imagenet/ eval.ckpt=checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar  # 80.6% acc
python run.py experiment=imagenet-t2tvit-eval.yaml model/t2tattn_cfg=performer datamodule.data_dir=/path/to/imagenet/ eval.ckpt=checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar  # 77.8-79.0% acc (there's randomness)
python run.py experiment=imagenet-t2tvit-eval.yaml model/t2tattn_cfg=sblocal datamodule.data_dir=/path/to/imagenet/ eval.ckpt=checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar  # 81.1% acc
```






