from matplotlib import pyplot as plt
import seaborn as sns
from sys import argv
import pandas as pd
import numpy as np

sns.set_theme()
for arg in argv[1:]:
    if "test" in arg:
        d = []
        with open(arg, "r") as f:
            [d.append(float(l[:-1])) for l in f]
        d = d[1:]
        sns.lineplot(data=d)
    elif "everything" in arg:
        df = pd.read_csv(arg, header=None)
        sns.lineplot(data=df.std(axis=1), errorbar="ci")
    else:
        df = pd.read_csv(arg)
        sns.regplot(x=np.arange(20000), y=df['avg'], line_kws={'color': 'C1'})
plt.show()
