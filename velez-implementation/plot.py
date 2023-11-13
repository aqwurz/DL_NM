from matplotlib import pyplot as plt
import seaborn as sns
from sys import argv
import pandas as pd
import numpy as np

sns.set_theme()
x = np.linspace(0, 2000, 2000)
y = (-0.087462*x-0.84)/(-0.09*x-1.2)
sns.lineplot(x=x, y=y, errorbar=None)
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
    else:
        df = pd.read_csv(arg, header=None)
        sns.lineplot(data=df.max(axis=1), errorbar=None)
plt.legend(labels=["expected"]+argv[1:])
plt.show()
