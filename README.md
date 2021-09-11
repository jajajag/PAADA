# Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation

The code is modified from [mixreg](https://github.com/kaixin96/mixreg) repo. This repo contains code for our proposed method Policy-Aware Adversarial Data Augmentation (PAADA). Code for PPO is based on [train-procgen](https://github.com/openai/train-procgen).

This code is only used for testing purpose. It may suffer from low efficiency. If you intend to do further research in this topic, we strongly suggest you to implement your only code.

## Requirements
The code can be run on a GPU with CUDA 9 or CUDA 10. We suggest you to run the code on a GPU with [CUDA 10](https://developer.nvidia.com/cuda-10.0-download-archive) and [cuDNN 10](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz). To install all the required dependencies, run

```
conda env create --file py37_cu10_tf115.yml
conda activate paada
```

## Experiments & results

Check out [experiments README](https://github.com/kaixin96/mixreg/blob/master/experiments/README.md) for running different experiments. You may also use the scripts in `experiments` folder to start training. All results are available at [Google Drive](https://drive.google.com/drive/folders/1wTURCswt6IfTDbEkBqMaIZhBlO7n8qDb?usp=sharing).
