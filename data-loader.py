import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.close("all")

data = pd.read_csv("data.csv", header=None)
data = data.drop(columns=[11])

values = data.values
values = (values - values.min(0)) / (values.max(0) - values.min(0))

ts = pd.Series(values[:, 1])
plt.figure()
plt.plot(ts)
plt.show()
