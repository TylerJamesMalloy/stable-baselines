# Capacity-Limited Actor-Critic Repository 

These files include the code for the Capacity-Limited Actor-Critic model using the Stable Baseline repository as a template for the code. 
Specifically this code is based promarily on the Soft Actor-Critic model. 

.. _clac:

.. automodule:: stable_baselines.clac


CLAC
===

`Capacity-Limited Reinforcement Learning: Applications in Deep Actor-Critic Methods for Continuous Control.

CLAC is an Deep Off-Policy Actor-Critic implementation of Capacity-Limited Reinforcement Learning, here presented as supplementary code alongside 
a paper submission to ICLR 2020. 


.. warning::

  The CLAC model does not support ``stable_baselines.common.policies`` because it uses double q-values
  and value estimation, as a result it must use its own policy models (see :ref:`clac_policies`).


.. rubric:: Available Policies

.. autosummary::
    :nosignatures:

    MlpPolicy
    LnMlpPolicy
    CnnPolicy
    LnCnnPolicy

Notes
-----

- Original paper: In Preperation for ICLR

.. note::

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

.. code-block:: python

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

Parameters
----------

.. autoclass:: CLAC
  :members:
  :inherited-members:

.. _sac_policies:

CLAC Policies
-------------

.. autoclass:: MlpPolicy
  :members:
  :inherited-members:


.. autoclass:: LnMlpPolicy
  :members:
  :inherited-members:


.. autoclass:: CnnPolicy
  :members:
  :inherited-members:


.. autoclass:: LnCnnPolicy
  :members:
  :inherited-members:


Custom Policy Network
---------------------

Similarly to the example given in the `examples <../guide/custom_policy.html>`_ page.
You can easily define a custom architecture for the policy network:

.. code-block:: python

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
