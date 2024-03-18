from matplotlib import pyplot as plt
import seaborn as sns
from sys import argv
import pandas as pd
import numpy as np

sns.set_theme()

x = range(20000)
y = []
with open('../../DBNM/PlotData/2017_01_29_Diff_Mod0_SeaF_PS_SIB_TO_CUT2_HOLE_RL05_PR75_Sig050_L/mmm001_2017-01-29_20_05_21_12391/modularity.dat',
          'r') as f:
    for _ in range(20000):
        y.append(float(f.readline().split()[1]))
sns.lineplot(x=x, y=y, errorbar=None)
dfs = []
for arg in argv[1:]:
    if "test" in arg:
        d = []
        with open(arg, "r") as f:
            [d.append(float(l[:-1])) for l in f]
        d = d[1:]
        sns.lineplot(data=d)
    elif "legacy" in arg:
        df = pd.read_csv(arg)
        sns.regplot(x=np.arange(20000), y=df['avg'], line_kws={'color': 'C1'})
        dfs.append(df)
    else:
        df = pd.read_csv(arg, header=None)
        #sns.lineplot(data=df.max(axis=1), errorbar=None)
        dfs.append(df)
sns.lineplot(data=[
    np.mean([d.loc[i].max() for d in dfs])
    for i in range(len(dfs[0]))], errorbar=None)
#plt.legend(labels=["expected"]+argv[1:]+["mean"])
plt.legend(labels=["expected", argv[1]])
plt.show()
