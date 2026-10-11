import matplotlib.pyplot as plt
import numpy as np
x = np.array([1,2,3,4,5,6,7])

y1 = np.array([15,25,35,46,57,45,98])
y2 = np.array([35,24,45,16,27,75,48])
y3 = np.array([85,74,75,26,-7,15,18])


style1 = dict(marker =".",markersize=10,markerfacecolor="red",markeredgecolor="yellow",linestyle="dotted")
style2 = dict(marker =".",markersize=10,markerfacecolor="green",markeredgecolor="black",linestyle="dashed")
style3 = dict(marker =".",markersize=10,markerfacecolor="cyan",markeredgecolor="brown",linestyle="solid",color="violet")
plt.plot(x,y1,**style1)
plt.plot(x,y2,**style2)
plt.plot(x,y3,**style3)
plt.show()
