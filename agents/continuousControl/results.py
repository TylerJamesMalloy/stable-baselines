import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
sns.set(style="ticks", color_codes=True)
sns.set(font_scale=1.5)

Data  = pd.read_pickle("./results/data.pkl")

print(Data)

assert(False) 
fig, ax = plt.subplots(nrows=1, ncols=1, sharey=False, sharex=True)

#sns.lineplot(x="Timestep", y="Episode Reward", hue="Model", ax=ax, ci=68, data=Data)
sns.lineplot(x="Episode Number", y="Episode Reward", hue="Model", ax=ax, ci=68, data=Data)

fig.suptitle('Roboschool Ant Results')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
