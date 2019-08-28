from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing

import numpy as np 
np.seterr(all=None)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import warnings 
warnings.filterwarnings('ignore')
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC, CLAC, DDPG, PPO1, A2C, PPO2
from stable_baselines.clac.policies import MlpPolicy as clac_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

import roboschool

ENVIRONMENT_NAME = 'ContinuousNChain-v0'
TRAINING_STEPS = 10
TRAINING_TIMESTEPS = 10000
TESTING_TIMESTEPS = 10000
TRAINING_ITERATIONS = 5
CURRENT_ITERATION = 1
SAVE_AGENTS = False 
SAVE_FINAL_AGENT = True 
#TRAINING_MODELS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
TRAINING_MODELS = np.logspace(-2, 1, num=10)
POLICY_KWARGS = dict(layers=[256, 256])

#env = gym.make(ENVIRONMENT_NAME)
#env = DummyVecEnv([lambda: env])
#MAX_ENV_STEPS = 200

def test(model, model_string, ent_coef, env, randomization, timestep):
    obs = env.reset()
    training_reuslts = pd.DataFrame()
    episode_reward = 0
    
    for i in range(TESTING_TIMESTEPS):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards[0]

        if(dones[0]):
            env_name = env.unwrapped.envs[0].spec.id
            d = {'Episode Reward': episode_reward, 'ent_coef': ent_coef, 'Timestep': timestep, 'Env': env_name, 'Randomization': randomization, 'Model': model_string}
            training_reuslts = training_reuslts.append(d, ignore_index = True)

            if(randomization == 1):
                env.unwrapped.envs[0].randomize()
            if(randomization == 2):
                env.unwrapped.envs[0].randomize_extreme()
            
            episode_reward = 0
    
    return training_reuslts


def train(training_tag):
    env = gym.make(ENVIRONMENT_NAME)
    env = DummyVecEnv([lambda: env]) 
    data = pd.DataFrame()
    #env._max_episode_steps = MAX_ENV_STEPS

    if(isinstance(training_tag, float)):
        model = CLAC(clac_MlpPolicy, env, ent_coef=training_tag, verbose=1, policy_kwargs = POLICY_KWARGS)
        
        for step in range(TRAINING_STEPS):
            (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
            #data = data.append(learning_results, ignore_index=True)
            data = data.append(test(model, "CLAC" + str(training_tag), training_tag, env, False, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "CLAC" + str(training_tag), training_tag, env, 1, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "CLAC" + str(training_tag), training_tag, env, 2, (step + 1) * TRAINING_TIMESTEPS))
            
            file_tag = str(training_tag).replace(".", "p")
            if(SAVE_AGENTS):   
                model.save("models/CLAC_" + ENVIRONMENT_NAME + "_s" + str(step) + "_t" + str(file_tag) + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model
        step = 0
        

        model = SAC(sac_MlpPolicy, env, ent_coef=training_tag, verbose=1, policy_kwargs = POLICY_KWARGS)
        for step in range(TRAINING_STEPS):
            (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
            #data = data.append(learning_results, ignore_index=True)

            data = data.append(test(model, "SAC" + str(training_tag), training_tag, env, False, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "SAC" + str(training_tag), training_tag, env, 1, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "SAC" + str(training_tag), training_tag, env, 2, (step + 1) * TRAINING_TIMESTEPS))
            
            file_tag = str(training_tag).replace(".", "p")
            if(SAVE_AGENTS):   
                model.save("models/SAC_" + ENVIRONMENT_NAME + "_s" + str(step) + "_t" + str(file_tag) + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))
            
        env.reset()
        del model

    if(training_tag == "CLAC"):
        model = CLAC(clac_MlpPolicy, env, verbose=1, policy_kwargs = POLICY_KWARGS)

        for step in range(TRAINING_STEPS):
            (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
            
            #data = data.append(learning_results, ignore_index=True)

            data = data.append(test(model, "CLAC", "auto", env, False, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "CLAC", "auto", env, 1, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "CLAC", "auto", env, 2, (step + 1) * TRAINING_TIMESTEPS))

            if(SAVE_AGENTS):
                model.save("models/CLAC_" + ENVIRONMENT_NAME + "_s" + str(step) + "_auto" + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model
    
    if(training_tag == "SAC"):
        model = SAC(sac_MlpPolicy, env, verbose=1, policy_kwargs = POLICY_KWARGS)

        for step in range(TRAINING_STEPS):
            (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)

            #data = data.append(learning_results, ignore_index=True)

            data = data.append(test(model, "SAC", "auto", env, False, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "SAC", "auto", env, 1, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "SAC", "auto", env, 2, (step + 1) * TRAINING_TIMESTEPS))

            if(SAVE_AGENTS):
                model.save("models/SAC_" + ENVIRONMENT_NAME + "_s" + str(step) + "_auto" + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model
    
    if(training_tag == "DDPG"):
        # the noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        model = DDPG(DDPG_MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, policy_kwargs = POLICY_KWARGS)

        for step in range(TRAINING_STEPS):
            (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)

            #data = data.append(learning_results, ignore_index=True)

            data = data.append(test(model, "DDPG", None, env, False, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "DDPG", None, env, 1, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "DDPG", None, env, 2, (step + 1) * TRAINING_TIMESTEPS))
            
            if(SAVE_AGENTS):
                model.save("models/DDPG_" + ENVIRONMENT_NAME + "_s" + str(step) + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model

    if(training_tag == "PPO1"):
        model = PPO1(MlpPolicy, env, verbose=1, policy_kwargs = POLICY_KWARGS)

        for step in range(TRAINING_STEPS):
            model.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)

            data = data.append(test(model, "PPO1", training_tag, env, False, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "PPO1", training_tag, env, 1, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "PPO1", training_tag, env, 2, (step + 1) * TRAINING_TIMESTEPS))
            
            if(SAVE_AGENTS):
                model.save("models/PPO1_" + ENVIRONMENT_NAME + "_s" + str(step) + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model
    
    if(training_tag == "A2C"):
        model = A2C(MlpPolicy, env, verbose=1, policy_kwargs = POLICY_KWARGS)

        for step in range(TRAINING_STEPS):
            model.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)

            data = data.append(test(model, "A2C", training_tag, env, False, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "A2C", training_tag, env, 1, (step + 1) * TRAINING_TIMESTEPS))
            data = data.append(test(model, "A2C", training_tag, env, 2, (step + 1) * TRAINING_TIMESTEPS))
            
            if(SAVE_AGENTS):
                model.save("models/A2C_" + ENVIRONMENT_NAME + "_s" + str(step) + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model

    return data

    
if __name__ == "__main__":
    pool = ThreadPool(len(TRAINING_MODELS)) #ensure that the length of the training models does not exceed the cpu number: multiprocessing.cpu_count()
    all_results = pd.DataFrame()            

    for _ in range(TRAINING_ITERATIONS):
        my_array = TRAINING_MODELS
        results = pool.map(train, my_array)
        
        for result in results:
            all_results = all_results.append(result, ignore_index=True)
        
        CURRENT_ITERATION += 1

    all_results.to_pickle("results/ContinuousNChain.pkl")

    print(all_results)