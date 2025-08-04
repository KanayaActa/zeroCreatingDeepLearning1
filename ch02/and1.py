def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    y = w1 * x1 + w2 * x2
    if y >= theta:
        return 1
    else:
        return 0

if __name__ == "__main__":
    print(AND(0, 0))  # Output: 0
    print(AND(0, 1))  # Output: 0
    print(AND(1, 0))  # Output: 0
    print(AND(1, 1))  # Output: 1