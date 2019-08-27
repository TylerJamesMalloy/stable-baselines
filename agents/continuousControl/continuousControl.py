from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing

import numpy as np 
np.seterr(all=None)
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC, CLAC, DDPG, PPO2
from stable_baselines.clac.policies import MlpPolicy as clac_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

import roboschool

ENVIRONMENT_NAME = 'RoboschoolAnt-v1'
TRAINING_TIMESTEPS = 1000000
TESTING_TIMESTEPS = 20000
TRAINING_ITERATIONS = 5
CURRENT_ITERATION = 1
SAVE_AGENTS = True 
TRAINING_MODELS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, "DDPG", "PPO2", "CLAC", "SAC"]
#TRAINING_MODELS = ["SAC", "CLAC", "DDPG", "PPO2"]
POLICY_KWARGS = dict(layers=[256, 256])
TEST_GENERALIZATION = True

def test(model, model_string, ent_coef, env, randomization):
    obs = env.reset()
    training_reuslts = pd.DataFrame()
    episode_reward = 0
    
    for i in range(TESTING_TIMESTEPS):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards[0]

        if(dones[0]):
            env_name = env.unwrapped.envs[0].spec.id
            d = {'Episode Reward': episode_reward, 'ent_coef': ent_coef, 'Timestep': -1, 'Env': env_name, 'Randomization': randomization, 'Model': model_string}
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

    if(isinstance(training_tag, float)):
        model = CLAC(clac_MlpPolicy, env, ent_coef=training_tag, verbose=1, policy_kwargs = POLICY_KWARGS)
        (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)
        data = data.append(learning_results, ignore_index=True)

        data = data.append(test(model, "CLAC", training_tag, env, False))
        
        file_tag = str(training_tag).replace(".", "p")
        if(SAVE_AGENTS):   
            model.save("models/CLAC_" + ENVIRONMENT_NAME + "_t" + str(file_tag) + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))
        
        env.reset()
        del model

        model = SAC(sac_MlpPolicy, env, ent_coef=training_tag, verbose=1, policy_kwargs = POLICY_KWARGS)
        (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)
        data = data.append(learning_results, ignore_index=True)

        data = data.append(test(model, "SAC", training_tag, env, False))
        
        file_tag = str(training_tag).replace(".", "p")
        if(SAVE_AGENTS):   
            model.save("models/SAC_" + ENVIRONMENT_NAME + "_t" + str(file_tag) + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))
        
        env.reset()
        del model

    if(training_tag == "CLAC"):
        model = CLAC(clac_MlpPolicy, env, verbose=1, policy_kwargs = POLICY_KWARGS)
        (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)
        
        data = data.append(learning_results, ignore_index=True)

        data = data.append(test(model, "CLAC", "auto", env, False))

        if(SAVE_AGENTS):
            model.save("models/CLAC_" + ENVIRONMENT_NAME + "_auto" + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model
    
    if(training_tag == "SAC"):
        model = SAC(sac_MlpPolicy, env, verbose=1, policy_kwargs = POLICY_KWARGS)
        (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)

        data = data.append(learning_results, ignore_index=True)

        data = data.append(test(model, "SAC", "auto", env, False))

        if(SAVE_AGENTS):
            model.save("models/SAC_" + ENVIRONMENT_NAME + "_auto" + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model
    
    if(training_tag == "DDPG"):
        # the noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        model = DDPG(DDPG_MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
        (model, learning_results) = model.learn(total_timesteps=TRAINING_TIMESTEPS)

        data = data.append(learning_results, ignore_index=True)

        data = data.append(test(model, "DDPG", None, env, False))
        
        if(SAVE_AGENTS):
            model.save("models/DDPG_" + ENVIRONMENT_NAME + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

        env.reset()
        del model

    if(training_tag == "PPO2"):
        model = PPO2(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=TRAINING_TIMESTEPS)

        data = data.append(test(model, "PPO2", training_tag, env, False))
        
        if(SAVE_AGENTS):
            model.save("models/PPO2_" + ENVIRONMENT_NAME + "_i" + str(CURRENT_ITERATION) + "_ts" + str(TRAINING_TIMESTEPS))

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

    all_results.to_pickle("results/data.pkl")
