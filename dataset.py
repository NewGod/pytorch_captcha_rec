import torch
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import pylab
from model.test import Net 
from torch import optim, nn 
from tqdm import tqdm
from torch import Tensor

import string
characters = string.digits + string.ascii_uppercase

width, height, n_len, n_class = 180, 80, 4, len(characters)


def decode(y):
    s = [];
    for x in y:
        s.append(''.join([characters[i] for i in x.argmax(axis = -1)]))
    return s

# next(gen()) will return 32 datas
# X is [height, width, 3]
# XX is [3, height, width]
# y is [n_len * n_class]
# yy is [n_len] like ['A', 'B', 'C', 'D']
def data_generator(times = 100,batch_size=32):
    generator = ImageCaptcha(width=width, height=height)

    for _ in range(times):
        X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
        XX = np.zeros((batch_size, 3, height, width), dtype=np.uint8)
        y = np.zeros((batch_size, n_len, n_class), dtype=np.uint8)
        yy = np.zeros((batch_size, 4), dtype=np.uint8)
        s = []
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            X[i] = generator.generate_image(random_str)
            XX[i] = np.transpose(X[i], (2,0,1));
            s.append(random_str)
            for j, ch in enumerate(random_str):
                y[i][j][characters.find(ch)] = 1
                yy[i][j] = characters.find(ch)
        yield torch.from_numpy(XX).float(),torch.from_numpy(y).float()


if __name__ == '__main__':
    batch_size = 4
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    XX = np.zeros((batch_size, 3, height, width), dtype=np.uint8)
    y = np.zeros((batch_size, n_len, n_class), dtype=np.uint8)
    yy = np.zeros((batch_size, 4), dtype=np.uint8)
    generator = ImageCaptcha(width=width, height=height)

    for i in range(batch_size):
        random_str = ''.join([random.choice(characters) for j in range(n_len)])
        print(random_str)
        X[i] = generator.generate_image(random_str)
        XX[i] = np.transpose(X[i], (2,0,1));
        for j, ch in enumerate(random_str):
            y[i][j][characters.find(ch)] = 1
            yy[i][j] = characters.find(ch)

    print(yy)
    print(decode(y))
