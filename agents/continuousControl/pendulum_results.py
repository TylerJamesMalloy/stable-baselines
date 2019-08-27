import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
sns.set(style="ticks", color_codes=True)
sns.set(font_scale=1.5)

Data  = pd.read_pickle("./results/pendulum.pkl")

train = Data.loc[Data['Randomization'] == 0]
extr = Data.loc[Data['Randomization'] == 2]

print(Data)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True)

#sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", ax=ax, ci=68, data=Data)

#sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", ax=ax1, ci=68, data=train)
#sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", ax=ax2, ci=68, data=extr)

sns.barplot(x="Model", y="Episode Reward", ax=ax1, ci=68, data=train)
sns.barplot(x="Model", y="Episode Reward", ax=ax2, ci=68, data=extr)

fig.suptitle('Inverted Pendulum Results')

#ax1.get_legend().remove()
#ax2.get_legend().remove()
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.show()
#fig.savefig('gen_results.png',  figsize=(45,45))