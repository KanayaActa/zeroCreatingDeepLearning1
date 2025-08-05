import numpy as np

# def cross_entropy_error(y, t)
#     delta = 1e-7  # 小さな値を加えることで対数の計算でのゼロ除算を防ぐ
#     return -np.sum(t * np.log(y + delta))


# one-hotエンコーディングされたラベルと予測値のクロスエントロピー誤差を計算する関数
# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y + 1e-7)) / batch_size

# 
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size