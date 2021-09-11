# Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation

The code is modified from [mixreg](https://github.com/kaixin96/mixreg) repo. This repo contains code for our proposed method Policy-Aware Adversarial Data Augmentation (PAADA). Code for PPO is based on [train-procgen](https://github.com/openai/train-procgen).

This code is only used for testing purpose. 

## Requirements
The code can be run on a GPU with CUDA 9 or CUDA 10. The version we used was CUDA 10. To install all the required dependencies:

```
conda env create --file py37_cu10_tf115.yml
conda activate paada
```

The code was run on a single GPU with [CUDA 10](https://developer.nvidia.com/cuda-10.0-download-archive) and [cuDNN 10](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz).

## Experiments & results

Check out [experiments README](https://github.com/kaixin96/mixreg/blob/master/experiments/README.md) for running different experiments. You may also use the scripts in `experiments` folder to start training. All results are available at [Google Drive](https://drive.google.com/drive/folders/1wTURCswt6IfTDbEkBqMaIZhBlO7n8qDb?usp=sharing).

## Citation
```
@misc{wang2020improving,
      title={Improving Generalization in Reinforcement Learning with Mixture Regularization}, 
      author={Kaixin Wang and Bingyi Kang and Jie Shao and Jiashi Feng},
      year={2020},
      eprint={2010.10814},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
