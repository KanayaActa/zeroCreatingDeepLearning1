import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.where(x <= 0, 0, 1)

if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)

    plt.plot(x, y)
    plt.title("Step Function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()
