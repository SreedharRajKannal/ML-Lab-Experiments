import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([1,2,3,4,5,6,7]) #hrs studied

y1 = np.array([15,25,35,56,67,75,98])

x2 = np.array([2,4,5,6,7,8,10]) #hrs studied

y2 = np.array([12,23,55,66,77,95,98])

#style = dict(marker=".",markerfacecolor ="red",markeredgecolor="green",linestyle="dotted",color ="violet")

plt.title("Marks")
plt.scatter(x1,y1, color="Blue",s=20, label="Class A")

plt.scatter(x2,y1, color="Red",s=20,label ="Class B")
plt.xlabel("Hrs studies")
plt.ylabel("Marks")
plt.legend()
plt.show()