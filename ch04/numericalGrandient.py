import numpy as np
import matplotlib.pyplot as plt

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  # 値を元に戻す

    return grad

# 2変数関数
def function_2(x):
    return x[0]**2 + x[1]**2

if __name__ == "__main__":
    # メッシュを作る
    x = np.arange(-2.0, 2.5, 0.25)
    y = np.arange(-2.0, 2.5, 0.25)
    X, Y = np.meshgrid(x, y)

    # 勾配ベクトルを計算
    grad_x = np.zeros_like(X)
    grad_y = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            grad = numerical_gradient(function_2, np.array([X[i,j], Y[i,j]]))
            grad_x[i,j] = grad[0]
            grad_y[i,j] = grad[1]

    # 勾配ベクトル場を描画
    plt.figure(figsize=(7,7))
    plt.quiver(X, Y, -grad_x, -grad_y, angles="xy", color="#666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.title("Gradient Field of f(x, y) = x^2 + y^2")
    plt.show()
