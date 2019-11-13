import os, logging, time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import gym 
import stable_baselines
#import pybullet as p
#import pybullet_data

import numpy as np
import pandas as pd

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.clac.policies import MlpPolicy as CLAC_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC, CLAC


NUM_RESAMPLES = 10
NUM_TRAINING_STEPS = 50000
NUM_AGENTS = 10

results  = pd.DataFrame()

training = True 
testing = False 

for agent_step in range(NUM_AGENTS):
    env = gym.make('ContinuousNChain-v0')
    env = DummyVecEnv([lambda: env])

    model = SAC(MlpPolicy, env, verbose=1)
    
    resample_step = 0

    while(resample_step < NUM_RESAMPLES):
        (model, learning_results) = model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=100)

        learning_results.to_pickle("SAC_auto_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

        training_mean = np.mean(learning_results["Episode Reward"])
        env.env_method("resample") 

        resample_step += 1
    
    agent_step += 1

    del model 
    del env


for agent_step in range(NUM_AGENTS):
    env = gym.make('ContinuousNChain-v0')
    env = DummyVecEnv([lambda: env])

    model = CLAC(CLAC_MlpPolicy, env, mut_inf_coef=0.5, verbose=1)
    
    resample_step = 0

    while(resample_step < NUM_RESAMPLES):
        (model, learning_results) = model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=100)

        learning_results.to_pickle("CLAC_auto_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

        training_mean = np.mean(learning_results["Episode Reward"])
        env.env_method("resample") 

        resample_step += 1
    
    agent_step += 1

    del model 
    del env

