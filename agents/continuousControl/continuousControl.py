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

import roboschool

ENVIRONMENT_NAME = 'RoboschoolInvertedPendulum-v1'
TRAINING_TIMESTEPS = 200
TRAINING_ITERATIONS = 5
CURRENT_ITERATION = 1
SAVE_AGENTS = True 
TRAINING_MODELS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, "CLAC", "SAC"]
#TRAINING_MODELS = ["SAC"]
POLICY_KWARGS = dict(layers=[256, 256])
TEST_GENERALIZATION = True

def train(training_tag):
    trainingData = pd.DataFrame()
    env = gym.make(ENVIRONMENT_NAME)
    env = DummyVecEnv([lambda: env]) 

    if(isinstance(training_tag, float)):
        model = CLAC(clac_MlpPolicy, env, ent_coef=training_tag, verbose=1, policy_kwargs = POLICY_KWARGS)
        (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)
        trainingData = trainingData.append(learning_results, ignore_index=True)
        
        file_tag = str(training_tag).replace(".", "p")
        if(SAVE_AGENTS):   
            model.save("models/CLAC_" + ENVIRONMENT_NAME + "_t" + str(file_tag) + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))
        
        env.reset()
        del model

        model = SAC(sac_MlpPolicy, env, ent_coef=training_tag, verbose=1, policy_kwargs = POLICY_KWARGS)
        (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)
        trainingData = trainingData.append(learning_results, ignore_index=True)
        
        file_tag = str(training_tag).replace(".", "p")
        if(SAVE_AGENTS):   
            model.save("models/SAC_" + ENVIRONMENT_NAME + "_t" + str(file_tag) + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))
        
        env.reset()
        del model

    if(training_tag == "CLAC"):
        model = CLAC(clac_MlpPolicy, env, verbose=1, policy_kwargs = POLICY_KWARGS)
        (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)
        
        trainingData = trainingData.append(learning_results, ignore_index=True)

        if(SAVE_AGENTS):
            model.save("models/CLAC_" + ENVIRONMENT_NAME + "_auto" + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model
    
    if(training_tag == "SAC"):
        model = SAC(sac_MlpPolicy, env, verbose=1, policy_kwargs = POLICY_KWARGS)
        (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)

        trainingData = trainingData.append(learning_results, ignore_index=True)

        if(SAVE_AGENTS):
            model.save("models/SAC_" + ENVIRONMENT_NAME + "_auto" + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model
    
    return trainingData

    
if __name__ == "__main__":
    pool = ThreadPool(len(TRAINING_MODELS)) #ensure that the length of the training models does not exceed the cpu number: multiprocessing.cpu_count()
    all_results = pd.DataFrame()            

    for _ in range(TRAINING_ITERATIONS):
        my_array = TRAINING_MODELS
        results = pool.map(train, my_array)
        
        for result in results:
            all_results = all_results.append(result, ignore_index=True)
        
        CURRENT_ITERATION += 1

    all_results.to_pickle("results/data.pkl")
