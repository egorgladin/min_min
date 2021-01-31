import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt


def get_data(filename, size, seed=0):
    data = load_svmlight_file(filename)
    full_len = len(data[1])

    random.seed(seed)
    indices = random.sample(range(full_len), size)
    Z, t = data[0][indices], data[1][indices]

    Z = np.hstack((Z.toarray(), np.ones((size, 1))))
    scaler = StandardScaler()
    Z = scaler.fit_transform(Z)
    most_important = [20, 21, 22, 34, 49, 50]
    column_order = most_important + list(set(range(Z.shape[1])) - set(most_important))
    Z[:] = Z[:, column_order]

    t = np.array(t * 2 - 3)
    return Z, t


def plot_F(F, xs, ys, grad_evals, title, file_name, univar=False):
    Fs = [F(w) for w in xs] if univar else [F(x, y) for x, y in zip(xs, ys)]
    fig = plt.figure()
    plt.plot(grad_evals, Fs)
    plt.title(title)
    plt.savefig(file_name)
