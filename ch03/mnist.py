import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)

if __name__ == "__main__":
    print("Training data shape:", x_train.shape)  # Should be (60000, 784)
    print("Training labels shape:", t_train.shape)  # Should be (60000,)
    print("Test data shape:", x_test.shape)  # Should be (10000, 784)
    print("Test labels shape:", t_test.shape)  # Should be (10000,)
    print("First training image label:", t_train[0])  # Should print the label of the first training image