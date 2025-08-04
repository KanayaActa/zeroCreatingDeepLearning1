from or1 import OR
from and1 import AND
from nand import NAND

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
if __name__ == "__main__":
    print(XOR(0, 0))  # Output: 0
    print(XOR(0, 1))  # Output: 1
    print(XOR(1, 0))  # Output: 1
    print(XOR(1, 1))  # Output: 0