from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing

import numpy as np 
np.seterr(all=None)
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd

import roboschool
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC, CLAC
from stable_baselines.clac.policies import MlpPolicy as clac_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy

ENVIRONMENT_NAME = 'RoboschoolAnt-v1'
TRAINING_TIMESTEPS = 10000
TRAINING_ITERATIONS = 1
CURRENT_ITERATION = 1
SAVE_AGENTS = True 
TRAINING_MODELS = ["SAC"]
POLICY_KWARGS = dict(layers=[256, 256])

env = gym.make(ENVIRONMENT_NAME)
env = DummyVecEnv([lambda: env]) 

model = SAC(sac_MlpPolicy, env, verbose=1, policy_kwargs = POLICY_KWARGS)
(model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)
