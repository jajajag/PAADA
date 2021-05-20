import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation
    with shared parameters
    """

    def __init__(self, env, observations, latent, adv_gamma, estimate_q=False,
            vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env         
        RL environment

        observations    
        tensorflow placeholder in which the observations will be fed

        latent          
        latent state from which policy distribution parameters should be
        inferred

        vf_latent       
        latent state from which value function should be inferred (if None, then
        latent is used)

        sess            
        tensorflow session to run calculations in (if None, default session is 
        used)

        **tensors       
        tensorflow tensors for additional attributes such as state or mask
        """

        self.X = observations
        self.state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution
        # type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]

        # JAG: Computational graph for adversarial loss
        self.reward = tf.placeholder(tf.float32, shape=self.vf.shape)
        self.old_X = tf.placeholder(tf.float32, shape=self.X.shape)
        self.adv_gamma = adv_gamma

        # Remember the input has shape (128, 64, 64, 3), 128 is the num envs
        # We should keep the first dimension of loss -> the number of envs
        self.axes = tuple(range(1, len(self.X.shape)))
        # Minimize \nabla_s {\pi_\theta * Adv} + gamma * (obs - old_obs) ^ 2
        # = \nabla_s {\pi_\theta * (reward - V)} + gamma * (obs - old_obs) ^ 2

        # TODO: Check the sign of self.neglogp
        self.loss = -self.neglogp * (self.reward - self.vf) \
               + self.adv_gamma * tf.reduce_sum(
                       tf.square(self.X - self.old_X), self.axes)

        # Gradient of loss wrt observation
        self.grads = tf.gradients(ys=self.loss, xs=self.X)

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

    # JAG: Calculate gradient of policy wrt state (observation)
    def adv_gradient(self, obs, reward, old_obs):
        feed_dict = {
                self.X: adjust_shape(self.X, obs),
                self.reward: adjust_shape(self.reward, reward),
                self.old_X: adjust_shape(self.old_X, old_obs),
        }
        # For debugging purpose
        #a = self.sess.run(-self.neglogp * (self.reward - self.vf), feed_dict)
        #b = self.sess.run(self.adv_gamma * tf.square(
        #    tf.reduce_sum(self.X - self.old_X, self.axes)), feed_dict)
        #c = self.sess.run(self.loss, feed_dict)
        #print(a[64], b[64], c[64])
        #print(self.sess.run(self.grads, feed_dict)[0][0])

        return self.sess.run(self.grads, feed_dict)

def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with
                # value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

