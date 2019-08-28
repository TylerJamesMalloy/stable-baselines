import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
sns.set(style="ticks", color_codes=True)
sns.set(font_scale=1.5)

data  = pd.read_pickle("./results/ContinuousNChain.pkl")

print(data)

data_clac = data.loc[data["Model"].str.contains("CLAC")]
data_sac = data.loc[data["Model"].str.contains("SAC")]

data1_clac = data_clac.loc[data_clac['Randomization'] == 0.0]
data2_clac = data_clac.loc[data_clac['Randomization'] == 1.0]
data3_clac = data_clac.loc[data_clac['Randomization'] == 2.0]

data1_sac = data_sac.loc[data_sac['Randomization'] == 0.0]
data2_sac = data_sac.loc[data_sac['Randomization'] == 1.0]
data3_sac = data_sac.loc[data_sac['Randomization'] == 2.0]


#fig, ax = plt.subplots(nrows=3, ncols=1, sharey=False, sharex=True)
fig, axes = plt.subplots(nrows=3, ncols=2, sharey=False, sharex=True)

print(axes.shape)

(ax1, ax2, ax3) = axes[:,0]
(ax4, ax5, ax6) = axes[:,1]

#sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", ax=ax, ci=68, data=Data)

#sns.lineplot(x="Model", y="Episode Reward", hue="Model", ax=ax, ci=68, data=data1)
#sns.lineplot(x="Model", y="Episode Reward", hue="Model", ax=ax, ci=68, data=data2)
#sns.lineplot(x="Model", y="Episode Reward", hue="Model", ax=ax, ci=68, data=data3)

sns.barplot(x="ent_coef", y="Episode Reward", ax=ax1, ci=68, data=data1_clac)
sns.barplot(x="ent_coef", y="Episode Reward", ax=ax2, ci=68, data=data2_clac)
sns.barplot(x="ent_coef", y="Episode Reward", ax=ax3, ci=68, data=data3_clac)

ax1.set_title("Capacity-Limited Actor-Critic")
ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("Mutual Information Coefficient")

ax1.set_ylabel("")
ax2.set_ylabel("")
ax3.set_ylabel("")

sns.barplot(x="ent_coef", y="Episode Reward", ax=ax4, ci=68, data=data1_sac)
sns.barplot(x="ent_coef", y="Episode Reward", ax=ax5, ci=68, data=data2_sac)
sns.barplot(x="ent_coef", y="Episode Reward", ax=ax6, ci=68, data=data3_sac)

ax4.set_title("Soft Actor-Critic")
ax4.set_xlabel("")
ax5.set_xlabel("")
ax6.set_xlabel("Entropy Coefficient")

ax4.set_ylabel("")
ax5.set_ylabel("")
ax6.set_ylabel("")

fig.text(0.04, 0.5, 'Episode Reward', va='center', rotation='vertical')
fig.suptitle('Continuous N-Chain Environment Results')
#plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
