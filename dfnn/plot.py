from glob import glob
from sys import argv

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()
#sns.set_palette(['#2c7fb8', '#7fcdbb', '#edf8b1'])
sns.set_palette(['#1b9e77', '#7570b3', '#d95f02', '#000000'])

x = range(20000)
y = []
for f in glob("../../DBNM/PlotData/2017_01_29_Diff_Mod0_SeaF_PS_SIB_TO_CUT2_HOLE_RL05_Sig050/*/modularity.dat"):
    with open(f, 'rb') as bf:
        bf_list = []
        add = True
        for _ in range(20000):
            l = bf.readline()
            try:
                bf_list.append(float(l.split()[1]))
            except IndexError:
                add = False
        if add:
            y.append(bf_list)
"""
with open('../../DBNM/PlotData/2017_01_29_Diff_Mod0_SeaF_PS_SIB_TO_CUT2_HOLE_RL05_PR75_Sig050_L/mmm001_2017-01-29_20_05_21_12391/modularity.dat',
          'r') as f:
    for _ in range(20000):
        y.append(float(f.readline().split()[1]))
"""
sns.lineplot(x=x, y=np.mean(y, axis=0), errorbar=None)
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
data = []
for i in range(len(dfs[0])):
    try:
        data.append(np.mean([d.loc[i].max() for d in dfs]))
    except KeyError:
        continue
sns.lineplot(data=data, errorbar=None)
#plt.legend(labels=["expected"]+argv[1:]+["mean"])
plt.legend(labels=["Original implementation", "New implementation"])
plt.xlabel("Num. generations")
plt.ylabel("Max fitness")
plt.show()
