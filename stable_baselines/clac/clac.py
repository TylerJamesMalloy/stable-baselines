import sys
import time
import multiprocessing
from collections import deque
import warnings

import numpy as np
import tensorflow as tf
import pandas as pd
import math 
from scipy.stats import norm

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.clac.policies import CLACPolicy
from stable_baselines import logger

from scipy.stats import multivariate_normal

from gym.spaces import Discrete

def get_vars(scope):
    """
    Alias for get_trainable_vars

    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)


class CLAC(OffPolicyRLModel):
    """
    Capacity-Limited Actor-Critic (CLAC)
    Off-Policy Capacity Limited Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from the Soft Actor-Critic Implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/) and from the Stable-Baseliens implementation 
    (https://github.com/hill-a/stable-baselines/tree/master/stable_baselines/sac)

    Paper: In Preperation for ICML 2020

    :param policy: (CLACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param mut_inf_coef: (str or float) Mutual Information regularization coefficient. Controlling
        performance/generalization trade-off. Set it to 'auto' to learn it automatically (still in development)
        (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_inf: (str or float) target mutual information when learning mut_inf_coef (mut_inf_coef = 'auto')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on CLAC logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=1000000,
                 learning_rate_phi=2e-3, learning_starts=100, train_freq=1, batch_size=256,
                 tau=0.005, mut_inf_coef='auto', target_update_interval=1, coef_schedule=None,
                 gradient_steps=1, target_entropy='auto', verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None):

        super(CLAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=CLACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # Same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        self.mut_inf_coef = mut_inf_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma

        self.coef_schedule = coef_schedule
        self.init_mut_inf_coef = self.mut_inf_coef
        
        # Options for MI approximation and related parameters 
        self.learning_rate_phi = learning_rate_phi # Taken from MIRL paper, not altered 
        self.multivariate_mean = None
        self.multivariate_cov = None 

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_mut_inf_coef = None
        self.logp_phi = None
        self.logp_pi = None
        self.tf_logged_reward = float("-inf")

        self.auto_mut_inf_coef = False
        if not isinstance(self.mut_inf_coef, float):
            self.auto_mut_inf_coef = True

        self.action_history = None
        self.action_entropy = 1

        if _init_setup_model:
            self.setup_model()
    
    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.deterministic_action * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    
                    # If the action space is discrete we want 
                    if(isinstance(self.env.action_space, Discrete)):
                        self.action_history = np.zeros((self.env.action_space.n))
                        self.actions_ph = tf.placeholder(tf.float32, shape=(None, self.env.action_space.n), name='actions')
                    else:
                        self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape, name='actions')
                    
                    self.logp_phi = tf.placeholder(tf.float32, shape=(None, ), name='logp_phi')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    self.mut_inf_coef_tensor = tf.placeholder(tf.float32, shape= (), name='mut_inf_coef')

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probabilty of actions taken by the policy
                    _, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # self.logp_pi = logp_pi
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)

                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph, policy_out, 
                                                                        create_qf=True, create_vf=False, reuse=True)
                    
                    #phi_proba, log_phi_proba = self.policy_tf.make_marginal()
                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = np.prod(self.env.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # Automatic mutual information coefficient setting is not fully tested
                    if isinstance(self.mut_inf_coef, str) and self.mut_inf_coef.startswith('auto'):
                        # Default initial value of mut_inf_coef when learned
                        init_value = 1.0 
                        if '_' in self.mut_inf_coef:
                            init_value = float(self.mut_inf_coef.split('_')[1])
                            assert init_value > 0., "The initial value of mut_inf_coef must be greater than 0"

                        self.log_mut_inf_coef = tf.get_variable('log_mut_inf_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.mut_inf_coef = tf.exp(self.log_mut_inf_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.mut_inf_coef = float(self.mut_inf_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Targets for Q and V regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    mut_inf_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.mut_inf_coef, float):
                        mut_inf_coef_loss = -tf.reduce_mean(
                            # self.log_mut_inf_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                            # self.log_mut_inf_coef * tf.stop_gradient((-1 * (self.logp_phi - logp_pi)) - self.target_entropy))
                            self.log_mut_inf_coef * tf.stop_gradient(self.logp_phi - logp_pi - self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    #policy_kl_loss = tf.reduce_mean(self.mut_inf_coef * logp_pi - qf1_pi)
                    policy_kl_loss = tf.reduce_mean((-1 * self.mut_inf_coef_tensor * (self.logp_phi - logp_pi)) - qf1_pi)

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss

                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    # v_backup = tf.stop_gradient(min_qf_pi - self.mut_inf_coef * logp_pi)
                    # previous tests 
                    # v_backup = tf.stop_gradient(min_qf_pi - self.mut_inf_coef * (self.logp_phi - logp_pi))
                    # Minimzing mutual information  
                    v_backup = tf.stop_gradient(min_qf_pi + (self.mut_inf_coef_tensor * (self.logp_phi - logp_pi)))
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss
                    discrete_loss = policy_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    if(isinstance(self.env.action_space, Discrete)):
                        policy_train_op = policy_optimizer.minimize(discrete_loss, var_list=get_vars('model/pi'))
                    else:
                        policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = get_vars('model/values_fn')
                    source_params = get_vars("model/values_fn/vf")
                    target_params = get_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy', 'mut_inf_coef_loss', 'log_policy', 'log_marginal']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi, 
                                         self.entropy, policy_train_op, train_values_op]#, phi_train_op]

                        # Add entropy coefficient optimization operation if needed
                        if mut_inf_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                mut_inf_coef_op = entropy_optimizer.minimize(mut_inf_coef_loss, var_list=self.log_mut_inf_coef)
                                self.infos_names += ['mut_inf_coef']
                                self.step_ops += [mut_inf_coef_op, mut_inf_coef_loss, self.mut_inf_coef]
                        

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar('entropy', self.entropy)
                    if mut_inf_coef_loss is not None:
                        tf.summary.scalar('mut_inf_coef_loss', mut_inf_coef_loss)
                    tf.summary.scalar('mut_inf_coef', self.mut_inf_coef)
                    tf.summary.scalar('log_policy', tf.reduce_mean(logp_pi))
                    tf.summary.scalar('log_marginal', tf.reduce_mean(self.logp_phi))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('episode_reward', self.tf_logged_reward)

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        if(isinstance(self.env.action_space, Discrete)):
            batch_actions = batch_actions.reshape(self.batch_size, self.env.action_space.n)
        else:
            batch_actions = batch_actions.reshape(self.batch_size, self.env.action_space.shape[0])
        
        # Determine the logp_phi based on the current batch:
        if(isinstance(self.env.action_space, Discrete)):
            assert(False) # Not implemented
            # not correct for discrete actions
            action_count = [np.count_nonzero(batch_actions == action) for action in batch_actions]
            action_count = action_count / len(batch_actions)
            # assert all values are percentages in: action_count
            logp_phi = np.log(action_count)
        else:
            EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)

            #mu =  np.mean(batch_actions,axis=0)
            #cov = np.cov(batch_actions, rowvar=False) + (np.identity(self.env.action_space.shape[0]) * EPS)

            mu = self.multivariate_mean 
            cov = self.multivariate_cov

            if(len(mu) == 1):
                mu = mu[0]

            try:
                multivar = multivariate_normal(mu, cov)
                logp_phi = multivar.logpdf(batch_actions) # * -1 
                logp_phi = logp_phi.reshape(self.batch_size, )
            except:
                # Mutual infomration coefficient is too small to contribute anything
                logp_phi = np.zeros(self.batch_size, )

        mut_inf_coef = self.mut_inf_coef

        # If coinrunner environment
        #batch_obs = np.squeeze(batch_obs, axis=1)
        #batch_next_obs  = np.squeeze(batch_next_obs, axis=1)
        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate,
            self.logp_phi: logp_phi,
            self.mut_inf_coef_tensor: mut_inf_coef
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        
        #qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.log_mut_inf_coef is not None:
            mut_inf_coef_loss, mut_inf_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, mut_inf_coef_loss, mut_inf_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def sample(self, num_samples=1000):
        samples = [[],[],[],[],[]]
        for state in range(self.observation_space.n):
            mean = []

            for _ in range(num_samples):
                action = (self.predict(state)[0][0] - self.action_space.low) / (self.action_space.high - self.action_space.low)[0]
                mean.append(action[0])

            samples[state].append(np.mean(mean))
        
        return samples

    def run(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="CLAC", reset_num_timesteps=True, randomization=0):


        start_time = time.time()
        episode_rewards = [0.0]
        learning_results = pd.DataFrame()
        obs = self.env.reset()
        self.episode_reward = np.zeros((1,))
        ep_info_buf = deque(maxlen=100)
        n_updates = 0
        infos_values = []

        reward_data = pd.DataFrame()

        for step in range(total_timesteps):                
            if(isinstance(self.env.action_space, Discrete)):
                actions = list(range(self.env.action_space.n))
                action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                rescaled_action = np.random.choice(actions, 1, p = action)[0]
            else:
                action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                # Rescale from [-1, 1] to the correct bounds
                rescaled_action = action * np.abs(self.action_space.low)

            new_obs, reward, done, info = self.env.step(rescaled_action)

            act_mu, act_std = self.policy_tf.proba_step(obs[None])
            obs = new_obs

            # Retrieve reward and episode length if using Monitor wrapper
            # info = info[0]
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                ep_info_buf.extend([maybe_ep_info])

            if writer is not None:
                # Write reward per episode to tensorboard
                ep_reward = np.array([reward]).reshape((1, -1))
                ep_done = np.array([done]).reshape((1, -1))
                self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                    ep_done, writer, self.num_timesteps)

            episode_rewards[-1] += reward
            if done:
                if not isinstance(self.env, VecEnv):
                    obs = self.env.reset()

                    if(randomization == 1):
                        try:
                            for env in self.env.unwrapped.envs:
                                env.randomize()
                        except:
                            print("Trying to randomize an environment that is not set up for randomization, check environment file")
                            assert(False)

                    if(randomization == 2):
                        try:
                            for env in self.env.unwrapped.envs:
                                env.randomize_extreme()
                        except:
                            print("Trying to extremely randomize an environment that is not set up for randomization, check environment file") 
                            assert(False)

                Model_String = "CLAC"
                if not self.auto_mut_inf_coef:
                    Model_String = "CLAC " + str(self.init_mut_inf_coef)
                
                env_name = self.env.unwrapped.envs[0].spec.id

                mut_inf_coef = self.init_mut_inf_coef
                if(type(self.mut_inf_coef) == tf.Tensor or np.isnan(mut_inf_coef)):
                    mut_inf_coef = "auto"
                Model_String = "CLAC" + str(mut_inf_coef)
                d = {'Episode Reward': episode_rewards[-1], 'Coefficient': mut_inf_coef, 'Timestep': self.num_timesteps, 'Episode Number': len(episode_rewards) - 1, 'Env': env_name, 'Randomization': randomization, 'Model': "CLAC"}
                learning_results = learning_results.append(d, ignore_index = True)
                
                self.tf_logged_reward = episode_rewards[-1]

                episode_rewards.append(0.0)
                
        return (self, learning_results)

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="CLAC", reset_num_timesteps=True, randomization=0):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            learning_results = pd.DataFrame()
            obs = self.env.reset()
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []

            reward_data = pd.DataFrame()

            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if self.num_timesteps < self.learning_starts:
                    if(isinstance(self.env.action_space, Discrete)):
                        action = []
                        for _ in range(self.env.action_space.n):
                            action.append(1/self.env.action_space.n)
                        rescaled_action = self.env.action_space.sample()
                    else:
                        action = self.env.action_space.sample()
                        # No need to rescale when sampling random action
                        rescaled_action = action
                else: 
                    if(isinstance(self.env.action_space, Discrete)):
                        actions = list(range(self.env.action_space.n))
                        action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                        rescaled_action = np.random.choice(actions, 1, p = action)[0]
                    else:
                        action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                        # Rescale from [-1, 1] to the correct bounds
                        rescaled_action = action * np.abs(self.action_space.low)

                if(not isinstance(self.env.action_space, Discrete)):
                    assert action.shape == self.env.action_space.shape

                # If coinrunner environment
                # rescaled_action = np.array(rescaled_action, ndmin=1)

                new_obs, reward, done, info = self.env.step(rescaled_action)

                act_mu, act_std = self.policy_tf.proba_step(obs[None])

                if(len(act_std) == 1):
                    act_std = act_std[0]

                #print("ACT MU FROM PROBA STEP", act_mu)
                #print("ACT STD FROM PROBA STEP", act_std)
                if self.num_timesteps > self.learning_starts:
                    # Only update marginal approximation after learning starts is completed
                    if(self.multivariate_mean is None):
                        self.multivariate_mean = act_mu
                    else:
                        previous_mean = self.multivariate_mean
                        self.multivariate_mean = ((1 - self.learning_rate_phi) * self.multivariate_mean) + (self.learning_rate_phi * act_mu)
                    if(self.multivariate_cov is None):
                        self.multivariate_cov = np.diag(act_std)
                    else:
                        cov = (self.learning_rate_phi * np.diag(act_std) + (1 - self.learning_rate_phi) * self.multivariate_cov)
                        mom_1 = (self.learning_rate_phi * np.square(np.diag(act_mu))) + ((1 - self.learning_rate_phi) * np.square(np.diag(previous_mean)))
                        mom_2 = np.square((self.learning_rate_phi * np.diag(act_mu)) + (1 - self.learning_rate_phi)*np.diag(previous_mean))
                        self.multivariate_cov = cov + mom_1 - mom_2 

                    # Update Beta parameter if coef_schedule is set 
                    if(self.coef_schedule is not None and self.mut_inf_coef > 1e-12):
                        # (1 - a) B + a(1/L()) # Loss based update schdule, for later 
                        
                        # Currently using linear schedule: 
                        self.mut_inf_coef *= (1 - self.coef_schedule)
                        
                    """if(self.num_timesteps % 1000 == 0):
                        print("updated mut_inf_coef: ", self.mut_inf_coef, " at time step ", self.num_timesteps)"""

                # Store transition in the replay buffer.
                #print("adding action to replay buffer: ", action)
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                # info = info[0]
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        if self.num_timesteps < self.batch_size or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        for mb_info_val in mb_infos_vals:
                            for mb_info in mb_info_val:
                                if mb_info is not None:
                                    infos_values.append(np.mean(mb_info))
                        #infos_values = np.mean(mb_infos_vals, axis=0)

                episode_rewards[-1] += reward
                if done:
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()

                        if(randomization == 1):
                            try:
                                for env in self.env.unwrapped.envs:
                                    env.randomize()
                            except:
                                print("Trying to randomize an environment that is not set up for randomization, check environment file")
                                assert(False)

                        if(randomization == 2):
                            try:
                                for env in self.env.unwrapped.envs:
                                    env.randomize_extreme()
                            except:
                                print("Trying to extremely randomize an environment that is not set up for randomization, check environment file") 
                                assert(False)

                    Model_String = "CLAC"
                    if not self.auto_mut_inf_coef:
                        Model_String = "CLAC " + str(self.mut_inf_coef)
                    
                    env_name = self.env.unwrapped.envs[0].spec.id

                    mut_inf_coef = self.init_mut_inf_coef
                    if(type(self.mut_inf_coef) == tf.Tensor or np.isnan(mut_inf_coef)):
                        mut_inf_coef = "auto"
                    Model_String = "CLAC" + str(mut_inf_coef)
                    d = {'Episode Reward': episode_rewards[-1], 'Coefficient': mut_inf_coef, 'Timestep': self.num_timesteps, 'Episode Number': len(episode_rewards) - 1, 'Env': env_name, 'Randomization': randomization, 'Model': "CLAC"}
                    learning_results = learning_results.append(d, ignore_index = True)
                    
                    self.tf_logged_reward = episode_rewards[-1]

                    episode_rewards.append(0.0)
                    

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)
                            
                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                    logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            return (self, learning_results)

    def action_probability(self, observation, state=None, mask=None, actions=None):
        if actions is None:
            warnings.warn("Even thought CLAC has a Gaussian policy, it cannot return a distribution as it "
                          "is squashed by an tanh before being scaled and ouputed. Therefore 'action_probability' "
                          "will only work with the 'actions' keyword argument being used. Returning None.")
            return None

        observation = np.array(observation)

        warnings.warn("The probabilty of taken a given action is exactly zero for a continuous distribution.")

        return np.zeros((observation.shape[0], 1), dtype=np.float32)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)
        observation = observation.reshape((-1,) + self.observation_space.shape)

        if(isinstance(self.env.action_space, Discrete)):
            # could replace this with map apply 
            actions = []
            action_distributions = self.policy_tf.step(observation, deterministic=False)
            available_actions = list(range(self.env.action_space.n))

            for action_distribution in action_distributions:
                action = np.random.choice(available_actions, 1, p = action_distribution)[0]
                actions.append(action)
        else:
            actions = self.policy_tf.step(observation, deterministic=False)
            actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
            actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction


        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "multivariate_mean": self.multivariate_mean,
            "multivariate_cov": self.multivariate_cov,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "mut_inf_coef": self.mut_inf_coef if isinstance(self.mut_inf_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            "num_timesteps": self.num_timesteps, 
            #"replay_buffer": self.replay_buffer,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "coef_schedule": self.coef_schedule,
            "init_mut_inf_coef": self.init_mut_inf_coef
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)