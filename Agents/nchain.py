import os, logging, time, multiprocessing 

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


NUM_RESAMPLES = 5
NUM_TRAINING_STEPS = 50000
NUM_AGENTS = 16
COEFS = [0.06]

results  = pd.DataFrame()

training = True 
testing = False 

nchain_folder = "nchain"

def test_coef(coef):
    for agent_step in range(NUM_AGENTS):
        clac_env = gym.make('ContinuousNChain-v0')
        clac_env = DummyVecEnv([lambda: clac_env])

        #model = SAC(MlpPolicy, env, verbose=1)
        clac_model = CLAC(CLAC_MlpPolicy, clac_env, mut_inf_coef=coef, verbose=0)

        sac_env = gym.make('ContinuousNChain-v0')
        sac_env = DummyVecEnv([lambda: sac_env])

        #model = SAC(MlpPolicy, env, verbose=1)
        sac_model = SAC(MlpPolicy, sac_env, ent_coef=coef, verbose=0)
        
        resample_step = 0

        # Set both environments to the same hidden values 

        sac_env.env_method("resample") 
        hiddenValues = sac_env.env_method("getHiddenValues")[0]

        print("coef: ", coef, "agent_step: ", agent_step, hiddenValues, "\n")
        
        sac_env.env_method("setHiddenValues", hiddenValues)
        clac_env.env_method("setHiddenValues", hiddenValues)

        # Train for some number of resampling steps: 

        while(resample_step < NUM_RESAMPLES):
            (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

            learning_results.to_pickle(nchain_folder + "/results/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")
            clac_model.save(nchain_folder + "/models/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))

            (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

            learning_results.to_pickle(nchain_folder + "/results/SAC_"+ str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")
            sac_model.save(nchain_folder + "/models/SAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))

            # Set both environments to the same hidden values 

            sac_env.env_method("resample") 
            hiddenValues = sac_env.env_method("getHiddenValues")[0]
            
            sac_env.env_method("setHiddenValues", hiddenValues)
            clac_env.env_method("setHiddenValues", hiddenValues)

            #print("step", resample_step, " ", hiddenValues)

            resample_step += 1

        # Test performance on randomized environments:

        clac_generalization_means = []
        sac_generalization_means = []
        
        agent_step += 1

def test_agent(agent_step):  
    for coef in COEFS:
        clac_env = gym.make('ContinuousNChain-v0')
        clac_env = DummyVecEnv([lambda: clac_env])

        #model = SAC(MlpPolicy, env, verbose=1)
        clac_model = CLAC(CLAC_MlpPolicy, clac_env, mut_inf_coef=coef, verbose=0)

        sac_env = gym.make('ContinuousNChain-v0')
        sac_env = DummyVecEnv([lambda: sac_env])

        #model = SAC(MlpPolicy, env, verbose=1)
        sac_model = SAC(MlpPolicy, sac_env, ent_coef=coef, verbose=0)
        
        resample_step = 0

        # Set both environments to the same hidden values 

        sac_env.env_method("resample") 
        hiddenValues = sac_env.env_method("getHiddenValues")[0]

        print("coef: ", coef, "agent_step: ", agent_step, hiddenValues, "\n")
        
        sac_env.env_method("setHiddenValues", hiddenValues)
        clac_env.env_method("setHiddenValues", hiddenValues)

        # Train for some number of resampling steps: 

        while(resample_step < NUM_RESAMPLES):
            (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

            learning_results["Resample"] = str(resample_step)

            learning_results.to_pickle(nchain_folder + "/results/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")
            clac_model.save(nchain_folder +  "/models/CLAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))

            (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS)

            learning_results["Resample"] = str(resample_step)

            learning_results.to_pickle(nchain_folder +  "/results/SAC_"+ str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step) + ".pkl")
            sac_model.save(nchain_folder + "/models/SAC_" + str(coef).replace(".", "p") + "_" + str(agent_step) + "_" + str(resample_step))

            # Set both environments to the same hidden values 

            sac_env.env_method("resample") 
            hiddenValues = sac_env.env_method("getHiddenValues")[0]
            
            sac_env.env_method("setHiddenValues", hiddenValues)
            clac_env.env_method("setHiddenValues", hiddenValues)

            #print("step", resample_step, " ", hiddenValues)

            resample_step += 1

        # Test performance on randomized environments:

        clac_generalization_means = []
        sac_generalization_means = []

agents = np.linspace(0, NUM_AGENTS - 1, NUM_AGENTS, dtype="int")

def mp_handler():
    p = multiprocessing.Pool(len(agents))
    p.map(test_agent, agents)

if __name__ == '__main__':
    mp_handler()