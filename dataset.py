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
    for index in range(n_len * n_class):
        if (y[index] > 0):
            tt = index % n_class;
            s.append(characters[tt])
    return s

# next(gen()) will return 32 datas
# X is [height, width, 3]
# XX is [3, height, width]
# y is [n_len * n_class]
# yy is [n_len] like ['A', 'B', 'C', 'D']
def data_generator(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    XX = np.zeros((batch_size, 3, height, width), dtype=np.uint8)
    y = np.zeros((batch_size, n_len * n_class), dtype=np.uint8)
    yy = np.zeros((batch_size, 4), dtype=np.uint8)
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            X[i] = generator.generate_image(random_str)
            XX[i] = np.transpose(X[i], (2,0,1));
            for j, ch in enumerate(random_str):
                y[i][j*n_class + characters.find(ch)] = 1
                yy[i][j] = characters.find(ch)
        yield torch.from_numpy(XX).float(),torch.from_numpy(y).float()


#X, XX, y, yy = next(data_generator())
