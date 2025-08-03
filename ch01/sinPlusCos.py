import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, linestyle = "--", label='cos(x)')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sine and Cosine Waves")
plt.legend()
plt.grid()
plt.show()