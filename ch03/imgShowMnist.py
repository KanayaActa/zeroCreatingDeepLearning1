import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)

    # Display the first training image
    img = x_train[0].reshape(28, 28)
    label = t_train[0]
    print("Label:", label)
    img_show(img)  # Show the image