import numpy as np
from varag import varag
from combined_method import combined_method
from utils import get_data
from scipy.linalg import eigh
from scipy.special import expit
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from time import time


def experiment_combined(x_0, y_0, Z, t, n, Rx, Ry, S, reg, eps, K, newton_steps):
    m = Z.shape[0]
    n_features = Z.shape[1]
    assert n < n_features
    d = n_features - n
    Zx = Z[:, :d]
    Zy = Z[:, -n:]

    def F(x, y):
        w = np.vstack((x, y))
        exponent = -np.squeeze(Z @ w) * t
        return -np.log(expit(-exponent)).mean() + reg * np.linalg.norm(y)**2

    def dF_dx(x, y):
        w = np.vstack((x, y))
        numerator = -Zx.T * t  # shape (d, m)
        exponent = (Z @ w).T * t  # shape (1, m)
        grads = numerator * expit(-exponent)  # shape (d, m)
        return grads.mean(axis=1, keepdims=True)  # shape (d, 1)

    def dFi_dy(x, y, i):
        w = np.vstack((x, y))
        numerator = -t[i] * Zy[i:i+1].T
        exponent = t[i] * w.T @ Z[i:i+1].T
        return numerator * expit(-exponent) + 2 * reg * y

    def dF_dy_full(x, y):
        w = np.vstack((x, y))
        numerator = - Zy.T * t  # shape (n, m)
        exponent = (Z @ w).T * t  # shape (1, m)
        regularizer_grads = np.repeat(2 * reg * y, m, axis=1)  # shape (n, m)
        return numerator * expit(-exponent) + regularizer_grads  # shape (n, m)

    mu = 2 * reg
    L = eigh(Zy.T @ Zy / m + 2 * reg * np.eye(n), eigvals_only=True, subset_by_index=[n-1, n-1]).item()

    vaidya_params = {'d': d, 'Rx': Rx, 'eps': eps, 'K': K, 'newton_steps': newton_steps}
    varag_params = {'n': n, 'Ry': Ry, 'm': m, 'S': S, 'mu': mu, 'L': L}
    xs, ys, aux_evals = combined_method(x_0, y_0, dF_dx, dFi_dy, dF_dy_full, vaidya_params, varag_params)
    Fs = [F(x, y) for x, y in zip(xs, ys)]
    return aux_evals, Fs


def experiment_varag(w_0, Z, t, n, S, R, reg):
    m = Z.shape[0]
    n_features = Z.shape[1]
    assert n < n_features
    d = n_features - n

    def F(w):
        exponent = -np.squeeze(Z @ w) * t
        return -np.log(expit(-exponent)).mean() + reg * np.linalg.norm(w[d:])**2

    def oracle(w, i):
        numerator = -t[i] * Z[i:i+1].T
        exponent = t[i] * w.T @ Z[i:i+1].T
        y = w.copy()
        y[:d] = 0
        return numerator * expit(-exponent) + 2 * reg * y

    def oracle_full(w):
        numerator = - Z.T * t  # shape (d+n, m)
        exponent = (Z @ w).T * t  # shape (1, m)
        y = w.copy()
        y[:d] = 0
        regularizer_grads = np.repeat(2 * reg * y, m, axis=1)  # shape (d+n, m)
        return numerator * expit(-exponent) + regularizer_grads  # shape (n, m)

    mu = 0
    diag = np.append(np.zeros(d), np.ones(n))
    L = eigh(Z.T @ Z / m + 2 * reg * np.diag(diag), eigvals_only=True, subset_by_index=[n-1, n-1]).item()

    ws, grad_evals = varag(w_0, m, oracle, oracle_full, mu, L, S, R)
    Fs = [F(w) for w in ws]
    return grad_evals, Fs


def main():
    Ry = 100
    R = 100
    newton_steps = 5

    Z, t = get_data("madelon")
    for d in [20, 30]:
        n = Z.shape[1] - d
        np.random.seed(0)
        w_0 = np.random.randn(Z.shape[1], 1)
        x_0, y_0 = w_0[:-n], w_0[-n:]

        eps = 1e-4
        Rx = 150 if d == 30 else 100
        K = 5 if d == 30 else 4
        S = 31 if d == 30 else 25
        reg = 0.05
        S_aux = 8
        start = time()

        aux_evals, Fs = experiment_combined(x_0, y_0, Z, t, n, Rx, Ry, S_aux, reg, eps, K, newton_steps)
        fig, ax = plt.subplots()
        plt.plot(aux_evals, Fs, label="Комбинированный подход")

        grad_evals, Fs = experiment_varag(w_0, Z, t, n, S, R, reg)
        plt.plot(grad_evals, Fs, label="Varag")
        plt.yscale('log')
        fsize = 15
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel(r"$F(x,y)$", fontsize=fsize)
        plt.xlabel(r"Количество вычислений $\nabla_y F_i$", fontsize=fsize)
        plt.legend(prop={'size': fsize})
        ax.set_yticks([2, 4, 6, 10, 20, 30])
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        plt.grid()
        plt.savefig(f"plots/d{d} grid.png", bbox_inches='tight')
        print(f"Iteration took {time()-start:.3f} s")


if __name__=="__main__":
    main()
