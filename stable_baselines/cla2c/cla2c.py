import time

import gym
import numpy as np
import tensorflow as tf

from scipy.stats import entropy

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.cla2c.policies import ActorCriticPolicy, RecurrentActorCriticPolicy #, ActorCriticMarginal
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.schedules import Scheduler
from stable_baselines.common.tf_util import mse, total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean
from stable_baselines.common.misc_util import flatten_action_mask

def discount_with_dones(rewards, dones, gamma):
    """
    Apply the discount value to the reward, where the environment is not done

    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :return: ([float]) The discounted rewards
    """
    discounted = []
    ret = 0  # Return: discounted reward
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)  # fixed off by one bug
        discounted.append(ret)
    return discounted[::-1]


class CLA2C(ActorCriticRLModel):
    """
    The CLA2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param mut_inf_coef: (float) Entropy coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate
    :param alpha: (float)  RMSProp decay parameter (default: 0.99)
    :param epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
        (default: 1e-5)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, mut_inf_coef=0.1, ent_coef=0.01, max_grad_norm=0.5,
                 learning_rate=7e-4, alpha=0.99, epsilon=1e-5, lr_schedule='constant', mut_schedule='constant', verbose=0,
                 tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, marginal_kwargs=None, agent_id=0,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, marginal_type="window", marginal_reverse=False):

        self.n_steps = n_steps
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.mut_inf_coef = mut_inf_coef
        self.init_mut_coef = mut_inf_coef
        self.ent_coef=ent_coef + mut_inf_coef
        self.init_ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        self.mut_schedule = mut_schedule
        self.marginal_reverse = marginal_reverse
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.learning_rate_ph = None
        self.n_batch = None
        self.actions_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.learning_rate_schedule = None
        self.summary = None
        
        self.marginal_type = marginal_type

        self.logp_phi = None
        self.action_history = None
        self.running_marginal = None 
        self.masked_marginal_ph = None 

        self.marginal_entropy = None
        self.marginal_ph = None
        self.marginal_loss = None

        self.maw = MultiAgentWindow(env=env, model=self, n_steps=n_steps, gamma=gamma)
        self.agent_id = agent_id

        if(marginal_reverse == True):
            self.mut_inf_coef = 0

        super(CLA2C, self).__init__(policy=policy,  env=env, verbose=verbose,  # marginal=marginal, 
                                    requires_vec_env=True,_init_setup_model=_init_setup_model, 
                                    policy_kwargs=policy_kwargs, # marginal_kwargs=marginal_kwargs,
                                    seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_model()
    
    def reset_coef(self, mut_inf_coef):
        if(mut_inf_coef == -1):
            self.mut_inf_coef = self.init_mut_coef

        self.mut_inf_coef = mut_inf_coef
        self.ent_coef = self.init_ent_coef + mut_inf_coef
        
    def _make_runner(self) -> AbstractEnvRunner:
        return CLA2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)

    def _get_pretrain_placeholders(self):
        policy = self.train_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.actions_ph, policy.policy
        return policy.obs_ph, self.actions_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the CLA2C model must be an " \
                                                                "instance of common.cla2c.ActorCriticPolicy."
            
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                self.action_history = np.zeros(self.env.action_space.n)

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.n_batch = self.n_envs * self.n_steps

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps

                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                         n_batch_step, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs,
                                              self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)
                
                with tf.variable_scope("loss", reuse=False):
                    self.actions_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.actions_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
                    self.pg_loss = tf.reduce_mean(self.advs_ph * neglogpac)
                    self.vf_loss = mse(tf.squeeze(train_model.value_flat), self.rewards_ph)
                    # https://arxiv.org/pdf/1708.04782.pdf#page=9, https://arxiv.org/pdf/1602.01783.pdf#page=4
                    # and https://github.com/dennybritz/reinforcement-learning/issues/34
                    # suggest to add an entropy component in order to improve exploration.

                    self.marginal_ph = tf.placeholder(dtype=tf.float32, shape=(self.n_batch, None), name="marginal_ph")  # tf.placeholder(dtype="float32", shape=() , name="marginal_ph")                    
                    self.marginal_loss = mse(self.marginal_ph, train_model.policy_proba)                                 #tf.reduce_mean(self.marginal_ph * neglogpac) #
                    
                    # Get marginal entropy as calculated by running average of actions taken. 
                    self.running_marginal_entropy = tf.placeholder(tf.float32, shape=(), name='running_marginal_entropy')

                    # Get the masked marginal distributions
                    self.masked_marginal_ph = tf.placeholder(dtype=tf.float32, shape=(self.n_batch, None), name="masked_marginal_ph")

                    # If using a update schedule for updated mutual information coefficient, capture it here 
                    self.updated_mut_inf_coef = tf.placeholder(tf.float32, shape=(), name='updated_mut_inf_coef')
                    self.updated_ent_coef = tf.placeholder(tf.float32, shape=(), name='updated_ent_coef')

                    if(self.marginal_type == "window"):
                        #logits = tf.multiply(self.running_marginal_entropy, train_model.action_mask_ph)
                        logits = self.masked_marginal_ph
                    else:
                        logits = tf.multiply(self.marginal_ph, train_model.action_mask_ph)
                        
                    a_0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
                    exp_a_0 = tf.exp(a_0)
                    exp_a_0 = tf.multiply(exp_a_0, train_model.action_mask_ph)
                    z_0 = tf.reduce_sum(exp_a_0, axis=-1, keepdims=True)
                    p_0 = exp_a_0 / z_0
                    self.marginal_entropy = tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), axis=-1)
                    
                    # original loss: loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef
                    # v1 loss: self.pg_loss - ((self.marginal_entropy - self.entropy) * self.updated_mut_inf_coef) + self.vf_loss * self.vf_coef

                    marginal = self.marginal_entropy * self.updated_mut_inf_coef
                    entropy = self.entropy * self.updated_ent_coef
                    loss = self.pg_loss + (marginal - entropy) + self.vf_loss * self.vf_coef

                    #tf.summary.scalar('marginal entropy', self.marginal_entropy)
                    tf.summary.scalar('running marginal entropy', self.running_marginal_entropy)
                    tf.summary.scalar('updated_mut_inf_coef', self.updated_mut_inf_coef)
                    tf.summary.scalar('updated_ent_coef', self.updated_ent_coef)
                    #tf.summary.scalar('marginal', self.marginal_ph)
                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    #tf.summary.scalar('loss', loss)

                    self.params = tf_util.get_trainable_vars("model")

                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('marginal', tf.reduce_mean(self.marginal_ph))
                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_ph, decay=self.alpha,
                                                    epsilon=self.epsilon)
                self.apply_backprop = trainer.apply_gradients(grads)

                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                self.proba_step = step_model.proba_step
                self.value = step_model.value
                self.initial_state = step_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)

                self.summary = tf.summary.merge_all()

    def _train_step(self, obs, states, rewards, masks, actions, values, action_masks, update, writer=None):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param action_masks: (np.ndarray) Mask invalid actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            for action in actions:
                self.action_history[action] += 1
                self.running_marginal = self.action_history / np.sum(self.action_history)

        mut_inf_coef = self.mut_inf_coef 
        if(self.mut_schedule != 'constant'):
            if(self.marginal_reverse):
                if(self.mut_inf_coef < self.init_mut_coef):
                    mut_inf_coef += self.mut_schedule
            else:
                mut_inf_coef = self.mut_inf_coef * (1 - self.mut_schedule)

        self.mut_inf_coef = mut_inf_coef
        advs = rewards - values
        cur_lr = None
        for _ in range(len(obs)):
            cur_lr = self.learning_rate_schedule.value()
        assert cur_lr is not None, "Error: the observation input array cannon be empty" 

        masked_marginals = []
        for action_mask in action_masks:
            masked_marginal = self.running_marginal * action_mask
            masked_marginal /= np.sum(masked_marginal)
            masked_marginals.append(masked_marginal)

        marginal_nn = self.train_model.marginal_approximation(obs=obs, state=states, mask=action_masks, action_mask=action_masks)

        td_map = {  self.train_model.obs_ph: obs, 
                    self.actions_ph: actions, 
                    self.advs_ph: advs,
                    self.train_model.action_mask_ph: action_masks, 
                    self.marginal_ph : marginal_nn, 
                    self.rewards_ph: rewards, 
                    self.running_marginal_entropy : entropy(self.running_marginal),
                    self.masked_marginal_ph : masked_marginals,
                    self.updated_mut_inf_coef : self.mut_inf_coef,
                    self.updated_ent_coef : self.ent_coef,
                    self.learning_rate_ph: cur_lr}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, marginal_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.marginal_loss, self.entropy, self.apply_backprop],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * self.n_batch))
            else:
                summary, policy_loss, value_loss, marginal_loss, policy_entropy, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.marginal_loss, self.entropy, self.apply_backprop], td_map)
            writer.add_summary(summary, update * self.n_batch)

        else:
            policy_loss, value_loss, marginal_loss, policy_entropy, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.marginal_loss, self.entropy, self.apply_backprop], td_map)

        return policy_loss, value_loss, marginal_loss, policy_entropy

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="CLA2C", reset_num_timesteps=False):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()
            self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
                                                    schedule=self.lr_schedule)

            t_start = time.time()
            callback.on_training_start(locals(), globals())

            for update in range(1, total_timesteps // self.n_batch + 1):

                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)
                # unpack
                obs, states, rewards, masks, actions, values, ep_infos, true_reward, action_masks = rollout

                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                _, value_loss, marginal_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values, action_masks,
                                                                 self.num_timesteps // self.n_batch, writer)
                n_seconds = time.time() - t_start
                fps = int((update * self.n_batch) / n_seconds)

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)


                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, rewards)
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", self.num_timesteps)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("marginal_loss", float(marginal_loss))
                    logger.record_tabular("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.dump_tabular()

        callback.on_training_end()
        return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "mut_inf_coef": self.mut_inf_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

    def model(self):
        return self.model
    
    def train_step_multi(self, obs, states, dones, rewards, masks, actions, action_masks, update, infos, agent_id, writer=None, reset_num_timesteps=False):
        
        # Find out a way to not redo this every time 
        self._setup_learn()
        self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=self.n_steps, schedule=self.lr_schedule)

        if(agent_id == self.agent_id):
            # Add the training step to recent window, when full use as a training iteration on all networks. 
            actions, values, states, neglogop = self.step(obs, states, dones, action_mask=action_masks)
            self.maw.step(obs, states, values, rewards, dones, masks, actions, action_masks, update, infos)
        else:
            # What is known about the state that the other agent observes? 
            # List all possible states 
            # Which state is most likely in relation to their action
            # Update beliefs based on this. 
            return 

        self.num_timesteps += self.n_envs

        if(self.maw.is_full()):
            new_tb_log = self._init_num_timesteps(reset_num_timesteps)
            with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, "CLA2C", new_tb_log) \
                    as writer:
                    
                obs, states, rewards, masks, actions, values, ep_infos, true_reward, action_masks = self.maw.get_rollout()
                policy_loss, value_loss, marginal_loss, policy_entropy = self._train_step(obs, states, rewards, masks, actions, values, action_masks, self.num_timesteps // self.n_batch, writer)

            self.maw.clear()

        return 

##
class MultiAgentWindow():
    def __init__(self, env, model, n_steps, gamma):
        self.mb_obs = []
        self.mb_actions = []
        self.mb_values = []
        self.mb_dones = []
        self.mb_action_masks = []
        self.ep_infos = []
        self.mb_rewards = []
        self.mb_states = []

        self.gamma = gamma
        self.states = None

        self.env = env
        self.model = model
        n_envs = env.num_envs
        self.batch_ob_shape = (n_envs * n_steps,) + env.observation_space.shape
        self.obs = np.zeros((n_envs,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.n_steps = n_steps
        self.states = model.initial_state
        self.dones = [False for _ in range(n_envs)]
        self.callback = None  # type: Optional[BaseCallback]
        self.continue_training = True
        self.n_envs = n_envs

        self.action_masks = []
        
    
    def step(self, obs, states, values, rewards, dones, masks, actions, action_masks, update, infos):
        self.mb_obs.append(np.copy(obs))
        self.mb_actions.append(actions)
        self.mb_values.append(values) 
        self.mb_dones.append(dones)
        self.mb_action_masks.append(action_masks.copy())
        self.ep_infos = []
        self.mb_rewards.append(rewards)
        self.update = update

        self.obs = obs
        self.states = states
        self.dones = dones

        for info in infos:
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                self.ep_infos.append(maybe_ep_info)

            # Added for Hanabi Env
            try:
                env_action_mask = self.env.env_method("valid_actions")
            except:
                env_action_mask = info.get('action_mask')
            
            self.action_masks.append(flatten_action_mask(self.env.action_space, env_action_mask))
    
    def clear(self):
        self.mb_obs = []
        self.mb_actions = []
        self.mb_values = []
        self.mb_dones = []
        self.mb_action_masks = []
        self.ep_infos = []
        self.mb_rewards = []
    
    def len(self):
        return(len(self.mb_obs))
    
    def is_full(self):
        return len(self.mb_obs) == self.n_steps
    
    def get_rollout(self):
        mb_masks = []
        ep_infos  = []
        true_rewards = []

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_action_masks = [], [], [], [], [], []

        mb_states = self.states

        try:
            self.action_masks = self.env.env_method("valid_actions")
        except:
            pass # if this is not a hanabi environment we may not have this envirnoment method  
        
        self.mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(self.mb_obs, dtype=self.env.observation_space.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(self.mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(self.mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1)
        mb_values = np.asarray(self.mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(self.mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_action_masks = np.asarray(self.mb_action_masks, dtype=np.float32).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        true_rewards = np.copy(mb_rewards)
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)

            mb_rewards[n] = rewards

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        mb_action_masks = mb_action_masks.reshape(-1, *mb_action_masks.shape[2:])
        true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])


        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, ep_infos, true_rewards, mb_action_masks

        
class CLA2CRunner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5, gamma=0.99):
        """
        A runner to learn the policy of an environment for an cla2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        super(CLA2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma

    def _run(self):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_action_masks = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            # Added for hanabi environment 
            try:
                self.action_masks = self.env.env_method("valid_actions")
            except:
                pass # if this is not a hanabi environment we may not have this envirnoment method  

            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones, action_mask=self.action_masks)

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            mb_action_masks.append(self.action_masks.copy())
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            obs, rewards, dones, infos = self.env.step(clipped_actions)

            self.action_masks.clear()
            self.model.num_timesteps += self.n_envs

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)

                # Added for Hanabi Env
                try:
                    env_action_mask = self.env.env_method("valid_actions")
                except:
                    env_action_mask = info.get('action_mask')
                
                self.action_masks.append(flatten_action_mask(self.env.action_space, env_action_mask))

            if self.callback is not None:
                # Abort training early
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            self.states = states
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_action_masks = np.asarray(mb_action_masks, dtype=np.float32).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        true_rewards = np.copy(mb_rewards)
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        mb_action_masks = mb_action_masks.reshape(-1, *mb_action_masks.shape[2:])
        true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, ep_infos, true_rewards, mb_action_masks
