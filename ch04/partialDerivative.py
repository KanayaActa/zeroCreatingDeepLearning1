import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

def function_2(x, y):
    return x**2 + y**2  # 2変数関数

if __name__ == "__main__":
    # -5~5の範囲でメッシュを作る
    x = np.arange(-5.0, 5.0, 0.1)
    y = np.arange(-5.0, 5.0, 0.1)
    X, Y = np.meshgrid(x, y)   # X,Yは格子状の座標
    Z = function_2(X, Y)

    # 3Dプロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')  # サーフェス表示
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.show()
