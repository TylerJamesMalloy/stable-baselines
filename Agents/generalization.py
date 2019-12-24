import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import gym 
import stable_baselines
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.clac.policies import MlpPolicy as CLAC_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC, CLAC

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)


NUM_AGENTS = 10
NUM_RESAMPLES = 1
TAGS = ["0p0", "0p1", "0p2", "0p3", "0p4", "0p5"]
All_Data = pd.DataFrame()

for tag in TAGS:
    for agent_id in range(NUM_AGENTS):
        for resample_num in range(NUM_RESAMPLES):
            clac_env = gym.make('ContinuousNChain-v0')
            clac_env = DummyVecEnv([lambda: clac_env])

            sac_env = gym.make('ContinuousNChain-v0')
            sac_env = DummyVecEnv([lambda: sac_env])


            clac_model_name = "CLAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            clac_model_file = "nchain/models/" + clac_model_name

            sac_model_name = "SAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            sac_model_file = "nchain/models/" + sac_model_name

            sac_model = SAC.load(sac_model_file, env=clac_env)
            clac_model = CLAC.load(clac_model_file, env=sac_env)

            print("model name:", model_name)

            del clac_env
            del clac_model
            del sac_env 
            del sac_model
            