# Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation
The code is modified from [mixreg](https://github.com/kaixin96/mixreg) repo. This repo contains code for our proposed method Policy-Aware Adversarial Data Augmentation (PAADA). Code for PPO is based on [train-procgen](https://github.com/openai/train-procgen).

This code is only used for testing purpose. It may suffer from low efficiency. If you intend to do further research in this topic, we strongly suggest you to implement your only code.


## Requirements
The code can be run on a GPU with CUDA 9 or CUDA 10. We suggest you to run the code on a GPU with [CUDA 10](https://developer.nvidia.com/cuda-10.0-download-archive) and [cuDNN 10](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz). To install all the required dependencies, run

```
conda env create --file py37_cu10_tf115.yml
conda activate paada
```


## Experiments
To train pure PAADA with &nu; = 0.5 augmentation degree on 500 levels generalization and &xi; = 1 Climber environment, run

```
python -m train_procgen.train --num_levels 500 --gpus_id 0 --adv_epochs 3052 --env_name climber --adv_nenv 1 --adv_adv 0.5
```

To train PAADA+Mixup, run

```
python -m train_procgen.train --num_levels 500 --gpus_id 0 --adv_epochs 3052 --env_name climber --adv_nenv 1 --adv_adv 0.5 --mix_mode mixreg
```

To train PAADA with different augmentation degree &nu; = 0.5, adjust --adv\_adv parameter (--adv\_adv 0 means pure PPO)
```
python -m train_procgen.train --num_levels 500 --gpus_id 0 --adv_epochs 3052 --env_name climber --adv_nenv 1 --adv_adv 1
```

To train PAADA with limited environment &xi; = 0.5 and &xi; = 0.25, run
```
python -m train_procgen.train --num_levels 500 --gpus_id 0 --adv_epochs 3052 --env_name climber --adv_nenv 0.5 --adv_adv 0.5
python -m train_procgen.train --num_levels 500 --gpus_id 0 --adv_epochs 3052 --env_name climber --adv_nenv 0.25 --adv_adv 0.5
```

To train PAADA+Mixup with different &alpha; and &beta; in Beta distribution B(&alpha;, &beta;), run
```
python -m train_procgen.train --num_levels 500 --gpus_id 0 --adv_epochs 3052 --env_name climber --adv_nenv 0.25 --adv_adv 0.5 --mix_alpha 0.2 --mix_beta 1
```


## Results


## Citation
