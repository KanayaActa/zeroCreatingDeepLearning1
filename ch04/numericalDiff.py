import numpy as np
import matplotlib.pyplot as plt

# bad implementation of numerical differentiation
# def numerical_diff(f, x):
#     h = 10e-50  # step size
#     return (f(x + h) - f(x)) / (h)

def numerical_diff(f, x):
    h = 1e-4  # step size
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x**2 + 0.1 * x
if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.title("Function 1")
    plt.grid()
    plt.show()
    numerical_diff(function_1, 5.0)  # Example usage of numerical_diff
    numerical_diff(function_1, 10.0)  # Example usage of numerical_diff