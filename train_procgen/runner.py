"""
Runner wrapper to add augmentation
"""

import itertools

import numpy as np
from baselines.ppo2.runner import Runner, sf01

from .data_augs import Cutout_Color, Rand_Crop

class RunnerWithAugs(Runner):
    def __init__(self, *, env, model, nsteps, gamma, lam,
            # JAG: Set up adversarial parameters
            adv_mode, adv_steps, adv_lr, adv_mix, adv_thresh, adv_adv,
            data_aug="no_aug", is_train=True):
        super().__init__(env=env, model=model, nsteps=nsteps, gamma=gamma,
                lam=lam)
        self.data_aug = data_aug
        self.is_train = is_train

        # JAG: Set up adversarial parameters
        self.adv_mode = adv_mode
        self.adv_steps = adv_steps
        self.adv_lr = adv_lr
        self.adv_mix = adv_mix
        self.adv_thresh = adv_thresh
        self.adv_adv = adv_adv

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

        # JAG: Here we also save current state, though mostly they are None
        mb_states = [self.states]
        epinfos = []
        # For n in range number of steps (Sample minibatch)
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run
            # self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(
                    self.obs, S=self.states, M=self.dones)
            # JAG: Update mb_states list
            mb_states.append(self.states)
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
            # 1. The adv_mode is not in [adv_step, adv_epoch]
            # 2. We do not update observation at current step/epoch 
            '''
            if (self.adv_mode != 'replace' and self.adv_mode != 'combine') \
                    or update <= self.adv_thresh:
                continue

            obs = mb_obs[t].copy().astype(np.float32)
            # JAG: We use the complicated version of reward here
            reward = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal \
                    + self.gamma * self.lam * nextnonterminal * lastgaelam
            #reward = mb_rewards[t]

            import time
            tt = time.time()
            # TODO: Flatten the list, then do gradient to all batches
            # Gradient descent on observations
            for it in range(self.adv_steps):
                # Do gradient descent to the observations
                # Actually, we should compute values again after the last iter
                grads, values = self.model.adv_gradient(
                    obs, reward, mb_actions[t], mb_obs[t])
                obs -= self.adv_lr * np.array(grads[0])
            print('gradient time', time.time() - tt)

            if self.adv_mode == 'combine':
                t *= 2
            # Save the adversarial observation and values
            # Perform mixup here
            # adv_mix = 1 means we use adversarial samples
            # adv_mix = 0 means we use original samples
            mb_obs[t] = self.adv_mix * obs + (1 - self.adv_mix) * mb_obs[t]
            if self.adv_adv == 'new':
                # Use our up to date advantage
                mb_values[t] = values
            elif self.adv_adv == 'old':
                # Use original advantage
                mb_values[t] = mb_values[t]
            else:
                # Mix mode for advantage 'mix'
                mb_values[t] = self.adv_mix * values \
                        + (1 - self.adv_mix) * mb_values[t]
            # We choose with probability adv_mix is the value cannot be mixed
            #rand = np.random.random()
            #mb_actions[t] = actions if rand < self.adv_mix else mb_actions[t]
            #mb_states[t] = states if rand < self.adv_mix else mb_states[t]
            '''
        step = 2
        obs_raw, obs_flat = mb_obs.shape, [-1] + list(mb_obs.shape)
        adv_obs = np.asarray(mb_obs, dtype=np.float32).reshape(obs_flat)
        #adv_obs_old = np.asarray(mb_obs, dtype=np.float32).reshape(obs_flat)
        #adv_actions = np.asarray(mb_actions).reshape(obs_flat)
        #adv_rewards = np.asarray(mb_rewards, dtype=np.float32).reshape(obs_flat)
        for t in range(0, self.nsteps, step):
            start, end = t * self.nsteps, (t + step) * self.nsteps
            mb_start, mb_end = t, t + step
            for it in range(self.adv_steps):
                # Do gradient descent to the observations
                # Actually, we should compute values again after the last iter
                grads, values = self.model.adv_gradient(
                        adv_obs[start:end],
                        mb_rewards[mb_start:mb_end],
                        mb_actions[mb_start:mb_end],
                        mb_obs[mb_start:mb_end])
                print(grads[0].shape)
                adv_obs[start:end] -= self.adv_lr * np.array(grads[0])
                print(obs)

        mb_returns = mb_advs + mb_values

        # JAG: Keep original size of output if the epoch is below threshold
        if update <= self.adv_thresh:
            mb_obs = mb_obs[:self.nsteps]
            mb_returns = mb_returns[:self.nsteps]
            mb_dones = mb_dones[:self.nsteps]
            mb_actions = mb_actions[:self.nsteps]
            mb_values = mb_values[:self.nsteps]
            mb_neglogpacs = mb_neglogpacs[:self.nsteps]
        # If we are above the threshold and in combine mode
        # Shuffle the indices. Randomly select samples from both original data
        # and augmented data
        # TODO: Set coeff for selecting samples (0.5 : 0.5)
        elif self.adv_mode == 'combine':
            ori_len, adv_ind = self.nsteps // 2, self.nsteps - self.nsteps // 2
            ori_ind = [i for i in range(self.nsteps)]
            adv_ind = [i for i in range(self.nsteps, self.nsteps * 2)]
            np.random.shuffle(ori_ind)
            np.random.shuffle(adv_ind)
            # Generate new combined data with indices
            mb_obs = mb_obs[ori_ind] + mb_obs[adv_ind]
            mb_returns = mb_returns[ori_ind] + mb_returns[adv_ind]
            mb_dones = mb_dones[ori_ind] + mb_dones[adv_ind]
            mb_actions = mb_actions[ori_ind] + mb_actions[adv_ind]
            mb_values = mb_values[ori_ind] + mb_values[adv_ind]
            mb_neglogpacs = mb_neglogpacs[ori_ind] + mb_neglogpacs[adv_ind]

        if self.data_aug != 'no_aug' and self.is_train:
            self.aug_func.change_randomization_params_all()
            self.obs = self.aug_func.do_augmentation(obs)

        # JAG: Return mb_states[-1] instead of mb_states
        # The obs has shape 256 * (128, 64, 64, 3), where 128 is num_env
        # sf01 flatten obs to 256 * 128 * (64, 64, 3)
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values,
            mb_neglogpacs)), mb_states[-1], epinfos)
