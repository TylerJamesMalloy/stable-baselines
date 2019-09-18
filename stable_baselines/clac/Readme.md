# Capacity-Limited Actor-Critic Repository 

These files include the code for the Capacity-Limited Actor-Critic model using the Stable Baseline repository as a template for the code. 
Specifically this code is based promarily on the Soft Actor-Critic model. 

CLAC
===

`Capacity-Limited Reinforcement Learning: Applications in Deep Actor-Critic Methods for Continuous Control.

CLAC is an Deep Off-Policy Actor-Critic implementation of Capacity-Limited Reinforcement Learning, here presented as supplementary code alongside 
a paper submission to ICLR 2020. 


  The CLAC model does not support ``stable_baselines.common.policies`` because it uses double q-values
  and value estimation, as a result it must use its own policy models (see :ref:`clac_policies`).

    MlpPolicy
    LnMlpPolicy
    CnnPolicy
    LnCnnPolicy

Notes
-----

- Original paper: In Preperation for ICLR

    The default policies for CLAC differ a bit from others MlpPolicy: it uses ReLU instead of tanh activation,
    to match the original paper


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========

Example
-------

```python
  import gym
  import numpy as np

  from stable_baselines.clac.policies import MlpPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import CLAC

  env = gym.make('Pendulum-v0')
  env = DummyVecEnv([lambda: env])

  model = CLAC(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=50000, log_interval=10)
  model.save("sac_pendulum")

  del model # remove to demonstrate saving and loading

  model = SAC.load("sac_pendulum")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()
```

Parameters
----------
```python
def __init__(self, policy, env, gamma=0.99, learning_rate=1e-4, buffer_size=10000,
             learning_starts=100, train_freq=1, batch_size=256,
             tau=0.005, mut_inf_coef='auto', target_update_interval=1,
             gradient_steps=1, target_entropy='auto', verbose=0, tensorboard_log=None,
             _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):
             
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
    performance/generalization trade-off. Set it to 'auto' to learn it automatically
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
```

Custom Policy Network
---------------------

Similarly to the example given in the `examples <../guide/custom_policy.html>`_ page.
You can easily define a custom architecture for the policy network:

```python

  import gym

  from stable_baselines.sac.policies import FeedForwardPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import CLAC

  # Custom MLP policy of three layers of size 128 each
  class CustomCLACPolicy(FeedForwardPolicy):
      def __init__(self, *args, **kwargs):
          super(CustomCLACPolicy, self).__init__(*args, **kwargs,
                                             layers=[128, 128, 128],
                                             layer_norm=False,
                                             feature_extraction="mlp")

  # Create and wrap the environment
  env = gym.make('Pendulum-v0')
  env = DummyVecEnv([lambda: env])

  model = CLAC(CustomCLACPolicy, env, verbose=1)
  # Train the agent
  model.learn(total_timesteps=100000)
```
