import numpy as np
import matplotlib.pyplot as plt

# 数値勾配（多変数対応版）
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 元に戻す
    return grad

# 勾配降下法
def gradient_descent(f, x0, learning_rate=0.1, step_num=50):
    x = x0.copy()
    trajectory = [x.copy()]  # 推移を保存して可視化用にする
    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= learning_rate * grad
        trajectory.append(x.copy())
    return x, np.array(trajectory)

# 例: f(x, y) = x^2 + y^2
def example_function(x):
    return np.sum(x**2)

if __name__ == "__main__":
    initial_x = np.array([-3.0, 4.0])  # 初期位置を2次元にする
    optimized_x, trajectory = gradient_descent(example_function, initial_x, learning_rate=0.1, step_num=30)

    print("Optimized x:", optimized_x)
    print("Function value at optimized x:", example_function(optimized_x))

    # === 可視化 ===
    # 等高線のためのグリッド作成
    x = np.arange(-4.0, 4.5, 0.1)
    y = np.arange(-4.0, 4.5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # 等高線を描画
    plt.figure(figsize=(7,7))
    plt.contour(X, Y, Z, levels=[0.5,1,2,4,8,16], colors='gray')
    
    # 勾配降下の軌跡を描画
    plt.plot(trajectory[:,0], trajectory[:,1], 'o-', color='red')
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title("Gradient Descent Path on f(x, y)=x^2+y^2")
    plt.grid()
    plt.show()
