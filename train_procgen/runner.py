"""
Runner wrapper to add augmentation
"""
import itertools

import numpy as np
from baselines.ppo2.runner import Runner, sf01

from .data_augs import Cutout_Color, Rand_Crop


MAX_ITER = 50


class RunnerWithAugs(Runner):
    def __init__(self, *, env, model, nsteps, gamma, lam,
            # JAG: Set up adversarial parameters
            adv_epsilon, adv_lr, adv_ratio, adv_thresh,
            data_aug="no_aug", is_train=True):
        super().__init__(env=env, model=model, nsteps=nsteps, gamma=gamma,
                lam=lam)
        self.data_aug = data_aug
        self.is_train = is_train

        # JAG: Set up adversarial parameters
        self.adv_epsilon = adv_epsilon
        self.adv_lr = adv_lr
        self.adv_ratio = adv_ratio
        self.adv_thresh = adv_thresh

        if self.is_train and self.data_aug != "no_aug":
            if self.data_aug == "cutout_color":
                self.aug_func = Cutout_Color(batch_size=self.obs.shape[0])
            elif self.data_aug == "crop":
                self.aug_func = Rand_Crop(batch_size=self.obs.shape[0],
                        sess=model.sess)
            else:
                raise ValueError("Invalid value for argument data_aug.")
            self.obs = self.aug_func.do_augmentation(self.obs)

    def run(self, update=1):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions = [], [], []
        mb_values, mb_dones, mb_neglogpacs = [],[],[]

        mb_states = self.states
        epinfos = []
        # For n in range number of steps (Sample minibatch)
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run
            # self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(
                    self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            obs, rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

            if self.data_aug != 'no_aug' and self.is_train:
                self.obs[:] = self.aug_func.do_augmentation(obs)
            else:
                self.obs[:] = obs

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        # JAG: We do adversarial to nsteps * ratio samples
        adv_ind = [i for i in range(self.nsteps)]
        np.random.shuffle(adv_ind)
        adv_ind = adv_ind[:int(self.nsteps * self.adv_ratio['adv'])]

        # JAG: Create adversarial data
        adv_obs = mb_obs.copy()
        adv_returns = mb_returns.copy()
        adv_dones = mb_dones.copy()
        adv_actions = mb_actions.copy()
        adv_values = mb_values.copy()
        adv_neglogpacs = mb_neglogpacs.copy()
        adv_states = mb_states
        adv_epinfos = epinfos.copy()

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] \
                    + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta \
                    + self.gamma * self.lam * nextnonterminal * lastgaelam

            # JAG: The main part of gradient descent to the observation
            # Skip the adversarial process if
            # 1. We do not update observation at current epoch 
            # 2. We do not update current obs (We only modify ratio * nsteps)
            if update <= self.adv_thresh or t not in adv_ind:
                continue

            # JAG: We use the complicated version of reward here
            reward = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal \
                    + self.gamma * self.lam * nextnonterminal * lastgaelam
            #reward = mb_rewards[t]
            obs = mb_obs[t].copy().astype(np.float32)

            # JAG: We can flatten the list and do gradient to all batches
            # However, the memories of our machines are limited to do this
            # Gradient descent on observations
            for it in range(MAX_ITER):
                # Do gradient descent to the observations
                grads = self.model.adv_gradient(
                        obs, reward, mb_actions[t], mb_obs[t])
                grads = np.array(grads[0])
                obs -= self.adv_lr * grads
                # Exit the loop if the gradient is small enough
                if np.mean(np.abs(grads)) < self.adv_epsilon: break
            # Actually, we compute value again after the last iter
            value = self.model.value(obs)

            # Save the adversarial observation and value
            # Perform mixup here
            # adv_ratio = 1 means we use adversarial samples
            # adv_ratio = 0 means we use original samples
            adv_obs[t] = self.adv_ratio['obs'] * obs \
                    + (1 - self.adv_ratio['obs']) * mb_obs[t]
            adv_values[t] = self.adv_ratio['value'] * value \
                    + (1 - self.adv_ratio['value']) * mb_values[t]

        mb_returns = mb_advs + mb_values
        # JAG: Calculate returns for adversarial returns
        adv_advs = mb_advs.copy()
        adv_returns = adv_advs + adv_values

        if self.data_aug != 'no_aug' and self.is_train:
            self.aug_func.change_randomization_params_all()
            self.obs = self.aug_func.do_augmentation(obs)

        # The obs has shape 256 * (128, 64, 64, 3), where 128 is num_env
        # sf01 flatten obs to 256 * 128 * (64, 64, 3)
        # Then, mb_obs[a][b] will be mb_obs[b * 256 + a]
        return ((*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions,
                mb_values, mb_neglogpacs)), mb_states, epinfos),
                # JAG: Create a copy of augmented data
                (*map(sf01, (adv_obs, adv_returns, adv_dones, adv_actions,
                adv_values, adv_neglogpacs)), adv_states, adv_epinfos))
