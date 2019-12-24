import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)

NUM_RESAMPLES = 10
NUM_TRAINING_STEPS = 100000
NUM_AGENTS = 10
All_Data = pd.DataFrame()

for agent_step in range(NUM_AGENTS):
    resample_step = 0 
    while(resample_step < NUM_RESAMPLES):

        #clac_results = np.load("nchain/results/CLAC_0p2_" + str(agent_step) + "_" + str(resample_step) + ".pkl", allow_pickle=True)
        #sac_results = np.load("nchain/results/SAC_auto_" + str(agent_step) + "_" + str(resample_step) + ".pkl", allow_pickle=True)

        clac_data = pd.read_pickle("nchain/results/CLAC_0p1_" + str(agent_step) + "_" + str(resample_step) + ".pkl")
        sac_data = pd.read_pickle("nchain/results/SAC_auto_" + str(agent_step) + "_" + str(resample_step) + ".pkl")

        All_Data = All_Data.append(clac_data, sort=False)
        All_Data = All_Data.append(sac_data, sort=False)
        
        resample_step += 1 


sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", data=All_Data)
plt.show()