import numpy as np 
np.seterr(all=None)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import warnings 
warnings.filterwarnings('ignore')
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math 
from scipy.stats import norm
from scipy.stats import beta
sns.set(style="ticks", color_codes=True)
sns.set(font_scale=1.5)

plt.style.use('fivethirtyeight')

RESULTS_FOLDER = "./t100000/models/"
NUM_TRAINING_STEPS = 100000

import gym 

from stable_baselines import SAC, CLAC
from stable_baselines.clac.policies import MlpPolicy as clac_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

model_tags = ["0p2", "0p4", "0p6", "0p8", "1p0"]

env = gym.make("ContinuousNChain-v0")
env = DummyVecEnv([lambda: env])

fig, axes = plt.subplots(nrows=2, ncols=len(model_tags), sharey=True, sharex=True)
#fig, clac_axes = plt.subplots(nrows=1, ncols=len(model_tags), sharey=True, sharex=True)

clac_axes = axes[0,:]
sac_axes  = axes[1,:]

for (model_index, model_tag) in enumerate(model_tags):
    clac_model_path = RESULTS_FOLDER + "CLAC_ContinuousNChain-v0_s0_t" + model_tag + "_i1_ts" + str(NUM_TRAINING_STEPS) + ".pkl"
    clac_model = CLAC.load(clac_model_path, env=env)
    
    sac_model_path = RESULTS_FOLDER + "SAC_ContinuousNChain-v0_s0_t" + model_tag + "_i1_ts" + str(NUM_TRAINING_STEPS) + ".pkl"
    sac_model = SAC.load(sac_model_path, env=env)

    clac_sample = []
    sac_sample = []

    for _ in range(1000):
        for state in [0,1,2,3,4]: 
            clac_action = clac_model.predict([state])[0][0][0] 
            clac_action = (clac_action + 1)/(2) # Normalize the [-1,1] action to [0,1], gym required actions spaces to be symmetric.
            clac_sample.append(clac_action)


            sac_action = sac_model.predict([state])[0][0][0]
            sac_action = (sac_action + 1)/(2)
            sac_sample.append(sac_action)
    
    clac_sample = np.asarray(clac_sample)
    sac_sample = np.asarray(sac_sample)

    mu_clac = clac_sample.mean()
    std_clac = clac_sample.std()

    x_axis = np.arange(0, 1, 0.01)

    sns.lineplot(x_axis, norm.pdf(x_axis,mu_clac,std_clac), ax=clac_axes[model_index])
    sns.lineplot(x_axis, beta.pdf(x_axis,10,10), ax=clac_axes[model_index])

    mu_sac = clac_sample.mean()
    std_sac = clac_sample.std()

    sns.lineplot(x_axis, norm.pdf(x_axis,mu_sac,mu_sac), ax=sac_axes[model_index])
    sns.lineplot(x_axis, beta.pdf(x_axis,10,10), ax=sac_axes[model_index])

handles, labels = clac_axes[0].get_legend_handles_labels()
fig.legend(handles, ('Hidden Values', 'Learned Policy'), 'center left')
plt.show()
    

    




