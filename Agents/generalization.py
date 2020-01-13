import os, logging, time, multiprocessing

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

nchain_filename = "nchain"

NUM_AGENTS = 25
NUM_RESAMPLES = 1
NUM_GENERALIZATION_EPISODES = 100
tags = [0.02, 0.04, 0.06, 0.08, 0.1] #, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
tag_strings = []
for tag in tags:
    tag_strings.append(str(tag).replace(".", "p"))

agents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

def test_tag(tag):
    All_Data = pd.DataFrame()

    for agent_id in range(NUM_AGENTS):
        #print("agent_id: ", agent_id, " tag: ", tag)
        for resample_num in range(NUM_RESAMPLES):
            clac_env = gym.make('ContinuousNChain-v0')
            clac_env = DummyVecEnv([lambda: clac_env])

            sac_env = gym.make('ContinuousNChain-v0')
            sac_env = DummyVecEnv([lambda: sac_env])

            clac_model_name = "CLAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            clac_model_file = nchain_filename + "/models/" + clac_model_name

            sac_model_name = "SAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            sac_model_file = nchain_filename +  "/models/" + sac_model_name

            sac_model = SAC.load(sac_model_file, env=sac_env)
            clac_model = CLAC.load(clac_model_file, env=clac_env)

            sac_env.env_method("resample") 
            hiddenValues = sac_env.env_method("getHiddenValues")[0]

            sac_env.env_method("setHiddenValues", hiddenValues)
            clac_env.env_method("setHiddenValues", hiddenValues)

            for i in range(NUM_GENERALIZATION_EPISODES):
                clac_done = False 
                sac_done = False 
                sac_reward = 0
                clac_reward = 0

                sac_obs = sac_env.reset()
                while not sac_done:
                    action, _states = sac_model.predict(sac_obs)
                    sac_obs, rewards, dones, info = sac_env.step(action)
                    sac_done = dones[0]
                    sac_reward -= 1
                
                clac_obs = clac_env.reset()
                while not clac_done:
                    action, _states = clac_model.predict(clac_obs)
                    clac_obs, rewards, dones, info = clac_env.step(action)
                    clac_done = dones[0]
                    clac_reward -= 1
                
                clac_data = {"Model":"CLAC", "Coefficient": tag, "Reward": clac_reward}
                sac_data = {"Model":"SAC", "Coefficient": tag, "Reward": sac_reward}

                All_Data = All_Data.append(sac_data, ignore_index=True)
                All_Data = All_Data.append(clac_data, ignore_index=True)
    return All_Data

def test_agent(agent_id):
    All_Data = pd.DataFrame()

    #tags = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    tags = [0.02, 0.04, 0.06, 0.08, 0.1]
    tag_strings = []
    for tag in tags:
        tag_strings.append(str(tag).replace(".", "p"))

    for tag in tag_strings:
        #print("agent_id: ", agent_id, " tag: ", tag)
        for resample_num in range(NUM_RESAMPLES):
            clac_env = gym.make('ContinuousNChain-v0')
            clac_env = DummyVecEnv([lambda: clac_env])

            sac_env = gym.make('ContinuousNChain-v0')
            sac_env = DummyVecEnv([lambda: sac_env])

            clac_model_name = "CLAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            clac_model_file = nchain_filename + "/models/" + clac_model_name

            sac_model_name = "SAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            sac_model_file = nchain_filename +  "/models/" + sac_model_name

            sac_model = SAC.load(sac_model_file, env=sac_env)
            clac_model = CLAC.load(clac_model_file, env=clac_env)

            #sac_env.env_method("resample") 
            sac_env.env_method("randomize") 
            #sac_env.env_method("extreme_randomize") 
            hiddenValues = sac_env.env_method("getHiddenValues")[0]

            sac_env.env_method("setHiddenValues", hiddenValues)
            clac_env.env_method("setHiddenValues", hiddenValues)

            for i in range(NUM_GENERALIZATION_EPISODES):
                clac_done = False 
                sac_done = False 
                sac_reward = 0
                clac_reward = 0

                sac_obs = sac_env.reset()
                while not sac_done:
                    action, _states = sac_model.predict(sac_obs)
                    sac_obs, rewards, dones, info = sac_env.step(action)
                    sac_done = dones[0]
                    sac_reward -= 1
                
                clac_obs = clac_env.reset()
                while not clac_done:
                    action, _states = clac_model.predict(clac_obs)
                    clac_obs, rewards, dones, info = clac_env.step(action)
                    clac_done = dones[0]
                    clac_reward -= 1
                
                clac_data = {"Model":"CLAC", "Coefficient": tag, "Reward": clac_reward}
                sac_data = {"Model":"SAC", "Coefficient": tag, "Reward": sac_reward}

                All_Data = All_Data.append(sac_data, ignore_index=True)
                All_Data = All_Data.append(clac_data, ignore_index=True)
    return All_Data

def mp_handler():
    All_Data = pd.DataFrame()

    p = multiprocessing.Pool(len(agents))
    All_Data = All_Data.append(p.map(test_agent, agents), ignore_index = True)
    return All_Data

if __name__ == '__main__':
    
    start_time = time.time()
    
    All_Data = mp_handler()

    All_Data.to_pickle("nchain/Randomization_Results_10K_5N.pkl")
    #print(All_Data)

    print("--- %s seconds ---" % (time.time() - start_time))