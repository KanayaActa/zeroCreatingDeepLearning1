import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

if __name__ == "__main__":
    print("ReLU Function")
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.title("ReLU Function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.show()