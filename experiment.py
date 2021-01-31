import numpy as np
import datetime
from varag import varag
from combined_method import combined_method
from utils import get_data, plot_F
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from time import time


def experiment_combined(x_0, y_0, Z, t, n, Rx, Ry, S, sigma_sq, eps, K, newton_steps):
    m = Z.shape[0]
    n_features = Z.shape[1]
    assert n < n_features
    d = n_features - n
    Zx = Z[:, :d]
    Zy = Z[:, -n:]

    def F(x, y):
        w = np.vstack((x, y))
        exponents = -np.squeeze(Z @ w) * t
        return np.log(1 + np.exp(exponents)).mean() + np.linalg.norm(y)**2 / sigma_sq

    def dF_dx(x, y):
        w = np.vstack((x, y))
        numerator = -Zx.T * t  # shape (d, m)
        exponent = (Z @ w).T * t  # shape (1, m)
        grads = numerator / (1 + np.exp(exponent))  # shape (d, m)
        return grads.mean(axis=1, keepdims=True)  # shape (d, 1)

    def dFi_dy(x, y, i):
        w = np.vstack((x, y))
        z = Z[i:i+1].T
        zy = Zy[i:i+1].T
        return -t[i] * zy / (1 + np.exp(t[i] * w.T @ z)) + 2 * y / sigma_sq

    def dF_dy_full(x, y):
        w = np.vstack((x, y))
        numerator = - Zy.T * t  # shape (n, m)
        exponent = (Z @ w).T * t  # shape (1, m)
        regularizer_grads = np.repeat(2 * y / sigma_sq, m, axis=1)  # shape (n, m)
        return numerator / (1 + np.exp(exponent)) + regularizer_grads  # shape (n, m)

    mu = 2 / sigma_sq
    L = eigh(Zy.T @ Zy / m + 2 * np.eye(n) / sigma_sq, eigvals_only=True, subset_by_index=[n-1, n-1]).item()

    vaidya_params = {'d': d, 'Rx': Rx, 'eps': eps, 'K': K, 'newton_steps': newton_steps}
    varag_params = {'n': n, 'Ry': Ry, 'm': m, 'S': S, 'mu': mu, 'L': L}
    xs, ys, aux_evals = combined_method(x_0, y_0, dF_dx, dFi_dy, dF_dy_full, vaidya_params, varag_params)
    Fs = [F(x, y) for x, y in zip(xs, ys)]

    # file_name = f"plots/combined_{datetime.datetime.now().time()}.png"
    # title = f"eps={eps}, sigma_sq={sigma_sq}, Rx={Rx}, Ry={Ry}, S={S}"
    # plot_F(F, xs, ys, aux_evals, title, file_name)
    return aux_evals, Fs


def experiment_varag(w_0, Z, t, n, S, R, sigma_sq):
    m = Z.shape[0]
    n_features = Z.shape[1]
    assert n < n_features
    d = n_features - n

    def F(w):
        exponents = -np.squeeze(Z @ w) * t
        return np.log(1 + np.exp(exponents)).mean() + np.linalg.norm(w[d:])**2 / sigma_sq

    def oracle(w, i):
        z = Z[i:i+1].T
        y = w.copy()
        y[:d] = 0
        return -t[i] * z / (1 + np.exp(t[i] * w.T @ z)) + 2 * y / sigma_sq

    def oracle_full(w):
        numerator = - Z.T * t  # shape (d+n, m)
        exponent = (Z @ w).T * t  # shape (1, m)
        y = w.copy()
        y[:d] = 0
        regularizer_grads = np.repeat(2 * y / sigma_sq, m, axis=1)  # shape (d+n, m)
        return numerator / (1 + np.exp(exponent)) + regularizer_grads  # shape (n, m)

    mu = 0
    diag = np.append(np.zeros(d), np.ones(n))
    L = eigh(Z.T @ Z / m + 2 * np.diag(diag) / sigma_sq, eigvals_only=True, subset_by_index=[n-1, n-1]).item()

    ws, grad_evals = varag(w_0, m, oracle, oracle_full, mu, L, S, R)
    w_opt = ws[-1]
    Fs = [F(w) for w in ws]

    # file_name = f"plots/varag_{datetime.datetime.now().time()}.png"
    # title = f"sigma_sq={sigma_sq}, R={R}, S={S}"
    # plot_F(F, ws, None, grad_evals, title, file_name, univar=True)
    return grad_evals, Fs


def main():
    # don't affect
    Ry = 100
    R = 100
    K = 15
    S = 40

    # fix for now
    newton_steps = 5
    m = 10000
    n = 45

    # eps = 5*1e-3
    # sigma_sq = 10.
    # Rx = 100
    # S_aux = 5
    Z, t = get_data("covtype.libsvm.binary.scale", m)

    # x_0 = np.zeros((Z.shape[1] - n, 1))
    # np.random.seed(0)
    # y_0 = np.random.randn(n, 1)
    # w_0 = np.vstack((x_0, y_0))
    np.random.seed(0)
    w_0 = np.random.randn(Z.shape[1], 1)
    x_0, y_0 = w_0[:-n], w_0[-n:]
    sigma_sq = 1e15

    for eps in [5*1e-3]: #, 1e-3
        for Rx in [10]: #7, 10, 20
            # for sigma_sq in [0.5, 1]:
            for S_aux in [6]: #4, 6, 8
                start = time()

                # aux_evals, Fs = experiment_combined(x_0, y_0, Z, t, n, Rx, Ry, S_aux, sigma_sq, eps, K, newton_steps)
                fig = plt.figure()
                # plt.plot(aux_evals, Fs, label="combined")

                grad_evals, Fs = experiment_varag(w_0, Z, t, n, S, R, sigma_sq)
                plt.plot(grad_evals, Fs, label="varag")
                plt.title(f"eps={eps}, Rx={Rx}, sigma_sq={sigma_sq}, S_aux={S_aux}")
                plt.legend()
                plt.savefig(f"plots/grid_search/n_30/{eps} {Rx} {sigma_sq} {S_aux}.png")
                print(f"Iteration took {time()-start:.3f} s")


if __name__=="__main__":
    main()
