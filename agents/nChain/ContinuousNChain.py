from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing

import numpy as np 
np.seterr(all=None)
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC, CLAC
from stable_baselines.clac.policies import MlpPolicy as clac_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy

#import roboschool

ENVIRONMENT_NAME = 'ContinuousNChain-v0'
TRAINING_TIMESTEPS = 5000
POLICY_KWARGS = dict(layers=[256, 256])

env = gym.make(ENVIRONMENT_NAME, alpha=10, beta=10, action_power = 6, max_step = 25)
env = DummyVecEnv([lambda: env]) 

model = CLAC(clac_MlpPolicy, env, verbose=1,  policy_kwargs = POLICY_KWARGS)
(model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=50)
print(learning_results)