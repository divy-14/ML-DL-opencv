import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("heart.csv")

# 1 for men and zero for women
p = "thal"
plt.scatter(df[p], df["target"])
plt.xlabel(p)
plt.ylabel("target")
plt.show()


