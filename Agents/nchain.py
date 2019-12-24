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


NUM_RESAMPLES = 1
NUM_TRAINING_STEPS = 500000
NUM_GEN_STEPS = 5000
NUM_GENERALIZATON_EPISODES = 20
NUM_AGENTS = 10

results  = pd.DataFrame()

training = True 
testing = False 

"""
Testing Block for N-Chain generalization:
"""

for agent_step in range(NUM_AGENTS):
    clac_env = gym.make('ContinuousNChain-v0')
    clac_env = DummyVecEnv([lambda: clac_env])

    #model = SAC(MlpPolicy, env, verbose=1)
    clac_model = CLAC(CLAC_MlpPolicy, clac_env, mut_inf_coef=0.3, verbose=1)

    sac_env = gym.make('ContinuousNChain-v0')
    sac_env = DummyVecEnv([lambda: sac_env])

    #model = SAC(MlpPolicy, env, verbose=1)
    sac_model = SAC(MlpPolicy, sac_env, ent_coef=0.3, verbose=1)
    
    resample_step = 0

    # Set both environments to the same hidden values 

    sac_env.env_method("resample") 
    hiddenValues = sac_env.env_method("getHiddenValues")[0]
    
    sac_env.env_method("setHiddenValues", hiddenValues)
    clac_env.env_method("setHiddenValues", hiddenValues)

    #print("step", agent_step, " ", hiddenValues)

    # Train for some number of resampling steps: 

    while(resample_step < NUM_RESAMPLES):
        (clac_model, learning_results) = clac_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=10000)

        learning_results.to_pickle("nchain/results/CLAC_0p3_" + str(agent_step) + "_" + str(resample_step) + ".pkl")
        clac_model.save("nchain/models/CLAC_0p3_" + str(agent_step) + "_" + str(resample_step))

        print("clac mean full ", np.mean(learning_results["Episode Reward"]))
        print("clac mean first 100 ", np.mean(learning_results["Episode Reward"][0:100]))
        print("clac mean last 100 ", np.mean(learning_results["Episode Reward"][-100:]))

        (sac_model, learning_results) = sac_model.learn(total_timesteps=NUM_TRAINING_STEPS, log_interval=10000)

        learning_results.to_pickle("nchain/results/SAC_0p3_" + str(agent_step) + "_" + str(resample_step) + ".pkl")
        sac_model.save("nchain/models/SAC_0p3_" + str(agent_step) + "_" + str(resample_step))

        print("sac mean full ", np.mean(learning_results["Episode Reward"]))
        print("sac mean first 100 ", np.mean(learning_results["Episode Reward"][0:100]))
        print("sac mean last 100 ", np.mean(learning_results["Episode Reward"][-100:]))

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

    generalization_results = pd.DataFrame()

    generalization_step = 0 
    while(generalization_step < NUM_GENERALIZATON_EPISODES):
        sac_env.env_method("resample")
        clac_env.env_method("resample")

        (clac_model, clac_learning_results) = clac_model.run(total_timesteps=NUM_GEN_STEPS)
        (sac_model, sac_learning_results) = sac_model.run(total_timesteps=NUM_GEN_STEPS)

        generalization_results = generalization_results.append(clac_learning_results)
        generalization_results = generalization_results.append(sac_learning_results)

        print("clac generaliation results ", np.mean(clac_learning_results["Episode Reward"]))
        print("sac generaliation results ", np.mean(sac_learning_results["Episode Reward"]))

        clac_generalization_means.append(np.mean(clac_learning_results["Episode Reward"]))
        sac_generalization_means.append(np.mean(sac_learning_results["Episode Reward"]))
        

        generalization_step += 1 
    
    generalization_results.to_pickle("nchain/results/Generalization_Results_" + str(agent_step) + "a_" + "_0p3.pkl")

    print(generalization_results)

    print("clac generaliation results TOTAL", np.mean(clac_generalization_means))
    print("sac generaliation results TOTAL", np.mean(sac_generalization_means))

    #print("sac hidden values: ", sac_env.env_method("getHiddenValues"))
    #print("clac hidden values: ", clac_env.env_method("getHiddenValues"))
    
    agent_step += 1





