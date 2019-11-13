import os, logging, time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import gym 
import stable_baselines
import pybullet as p
import pybullet_data

import numpy as np
import pandas as pd

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC


NUM_RESAMPLES = 10
NUM_TRAINING_STEPS = 25000
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
        (model, learning_results) = model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=50)
        
        print(learning_results)

        learning_results.to_pickle("SAC_auto_" + str(agent_step))

        training_mean = np.mean(learning_results["Episode Reward"])
        env.env_method("resample") 

        resample_step += 1
    
    agent_step += 1

    del model 
    del env

if(testing):
    model = SAC.load("nchain/agent")
    env.env_method("resample") 

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        if(dones[0]):
            print(rewards[0])

            env.env_method("resample") 

