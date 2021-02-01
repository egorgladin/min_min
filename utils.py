import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import random


def get_data(filename, size=None, seed=0):
    data = load_svmlight_file(filename)
    full_len = len(data[1])

    if size is None:
        Z, t = data[0], data[1]
        size = Z.shape[0]
    else:
        random.seed(seed)
        indices = random.sample(range(full_len), size)
        Z, t = data[0][indices], data[1][indices]

    Z = Z.toarray()
    scaler = StandardScaler()
    Z = scaler.fit_transform(Z)
    Z = np.hstack((Z, np.ones((size, 1))))
    return Z, t
