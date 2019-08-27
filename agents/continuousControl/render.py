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

import roboschool

# Environments 
"""
RoboschoolInvertedPendulum-v1
RoboschoolInvertedPendulumSwingup-v1
RoboschoolInvertedDoublePendulum-v1
RoboschoolHopper-v1
RoboschoolWalker2d-v1
RoboschoolHalfCheetah-v1
RoboschoolAnt-v1
RoboschoolHumanoid-v1
"""

ENVIRONMENT_NAME = 'RoboschoolHumanoid-v1'
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
