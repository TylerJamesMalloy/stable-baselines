import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)

nchain_filename = "nchain_250K"


"""
#All_Data = pd.read_pickle(nchain_filename + "/Generalization_Results_10K_5N.pkl")
All_Data = pd.read_pickle(nchain_filename + "/Randomization_Results_10K_5N.pkl")

print(All_Data)

#g = sns.FacetGrid(All_Data, col="Model", sharex=False)
#g.map(sns.boxplot, 'Coefficient', 'Reward')

ax = sns.boxplot(x="Coefficient", y="Reward", hue="Model", data=All_Data)  # RUN PLOT   
#ax.set_ylim(-150,-100)
plt.title('250K Time Step NChain Randomization Results')
plt.show()
"""

NUM_AGENTS = 24
NUM_RESAMPLES = 10
NUM_GENERALIZATION_EPISODES = 100

#tags = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
tags = [0.04]
tag_strings = []
for tag in tags:
    tag_strings.append(str(tag).replace(".", "p"))

All_Data = pd.DataFrame()

agents = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

for tag in tag_strings:
    for agent_id in agents:
        for resample_num in range(NUM_RESAMPLES):

            clac_model_name = "CLAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            clac_model_file = nchain_filename + "/results/" + clac_model_name + ".pkl"

            sac_model_name = "SAC" + "_" + str(tag) + "_" + str(agent_id) + "_" + str(resample_num) 
            sac_model_file = nchain_filename +  "/results/" + sac_model_name + ".pkl"
            
            #print(pd.read_pickle(sac_model_file))

            clac_data = pd.read_pickle(clac_model_file)
            sac_data = pd.read_pickle(sac_model_file)

            #clac_data["Coefficient"] = clac_data["mut_inf_coef"]
            #sac_data["Coefficient"] = sac_data["ent_coef"]

            #clac_data = clac_data.iloc[-10:]
            #sac_data = sac_data.iloc[-10:]

            clac_data.loc[clac_data['Model'].str.contains('CLAC'), 'Model'] = 'CLAC'
            sac_data.loc[sac_data['Model'].str.contains('SAC'), 'Model'] = 'SAC'

            All_Data = All_Data.append(clac_data, sort=False)
            All_Data = All_Data.append(sac_data, sort=False)

print(All_Data)

#sns.lineplot(x="Resample", y="Episode Reward", hue="Model", data=All_Data, ci="sd") 

#ax = sns.boxplot(x="Resample", y="Episode Reward", hue="Model", data=All_Data, whis=100000)  # RUN PLOT   

plt.title('250K Time Step NChain Resample Results')
plt.show()

All_Data = pd.DataFrame()