import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import datetime
from sklearn.datasets import load_svmlight_file

def get_H_inv(A, b, x):
    d = A.shape[1]
    H = np.zeros((d, d))
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        H += a @ a.T / (a.T @ x - b[i])**2
    return np.linalg.inv(H)

def get_sigmas(A, b, x, H_inv):
    sigmas = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        sigmas[i] = a.T @ H_inv @ a / (a.T @ x - b[i])**2
    return sigmas

def get_Q_inv(sigmas, A, b, x):
    d = A.shape[1]
    Q = np.zeros((d, d))
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        Q += sigmas[i] * a @ a.T / (a.T @ x - b[i])**2
    return np.linalg.inv(Q)

def get_dV(sigmas, A, b, x):
    dV = np.zeros((A.shape[1], 1))
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        dV -= sigmas[i] * a / (a.T @ x - b[i])**2
    return dV

def get_vol_center(A, b, x, n_steps=20, stepsize=0.18):
    for step in range(n_steps):
        H_inv = get_H_inv(A, b, x)
        sigmas = get_sigmas(A, b, x, H_inv)
        Q_inv = get_Q_inv(sigmas, A, b, x)
        dV = get_dV(sigmas, A, b, x)
        x = x - stepsize * Q_inv @ dV
    return x

def get_beta(A, c, x, eps, H_inv):
    denom = np.sqrt(eps) * c.T @ H_inv @ c
    beta = x.T @ c - np.sqrt(5. / denom)
    return beta

def add_row(A, b, c, beta):
    A = np.vstack((A, c.T))
    b = np.append(b, beta)
    return A, b

def remove_row(A, b, i):
    A = np.delete(A, i, 0)
    b = np.delete(b, i)
    return A, b

def vaidya(A_0, b_0, x_0, eps, K, oracle_full):
    """Use Vaidya's method to minimize f(x)."""
    A_k, b_k = A_0, b_0
    x_k = x_0
    vals = []
    for k in range(K):
        if k % 10 == 0:
            print(f"k={k}")
        x_k = get_vol_center(A_k, b_k, x_k)
        vals.append(f(x_k).item())
        H_inv = get_H_inv(A_k, b_k, x_k)
        sigmas = get_sigmas(A_k, b_k, x_k, H_inv)
        if (sigmas >= eps).all():
            c_k = -oracle_full(x_k)
            beta_k = get_beta(A_k, c_k, x_k, eps, H_inv)
            A_k, b_k = add_row(A_k, b_k, c_k, beta_k)
        else:
            i = sigmas.argmin()
            A_k, b_k = remove_row(A_k, b_k, i)
    return x_k, vals

def get_init_polytope(d, R):
    # Задать начальное множество A_0, b_0 для радиуса R
    A_0 = np.vstack((np.eye(d), -np.ones((1, d))))
    b_0 = -R * np.ones(d + 1)
    b_0[-1] *= d
    return A_0, b_0

def entropy_experiment():
    d = 2
    R = 30
    A_0, b_0 = get_init_polytope(d, R)
    # x_0 = np.zeros((d, 1))
    x_0 = -5 * np.ones((d, 1))
    eps = 5 * 1e-3
    K = 100

    def f(x):
        return (x.T + 45) @ np.log((x + 45) * np.exp(-1) / 50)

    def oracle_full(x):
        return np.log((x + 45) * np.exp(-1) / 50) + 1

    x_opt, vals = vaidya(A_0, b_0, x_0, eps, K, oracle_full)
    print(x_opt)
    plt.plot(vals)
    plt.savefig(f"plots/test_{datetime.datetime.now().time()}.png")


# ======================== VARAG ========================


def get_s_0(m):
    return int(np.log(m)) + 1


def get_params(s, s_0, L, mu, m):
    if s <= s_0:
        alpha = 0.5
        T = 2 ** (s-1)
    else:
        min_ = min(np.sqrt(m * mu / (3 * L)), 0.5)
        alpha = max(2 / (s - s_0 + 4), min_)
        T = 2 ** (s_0-1)

    p = 0.5
    gamma = 1 / (3 * L * alpha)
    return T, gamma, alpha, p


def get_y_underbar(mu, gamma, alpha, p, y_t_bar, y_t, y_tilde):
    denom = 1 + mu * gamma * (1 - alpha)
    term = (1 + mu * gamma) * (1 - alpha - p) * y_t_bar
    return (term + alpha*y_t + (1 + mu * gamma)*p*y_tilde) / denom


def get_G(grad_full, grad_tilde, grad):
    return grad - grad_tilde + grad_full


def argmin_y(gamma, G, mu, y_underbar, y_t, R):
    y = (y_t + mu * gamma * y_underbar - gamma * G) / (1 + mu * gamma)
    norm = np.linalg.norm(y)
    return y * R / norm if norm > R else y


def get_thetas(gamma, alpha, p, T, L, mu, s, s_0, m):
    if 1 <= s <= s_0 or (s_0 < s <= s_0 + np.sqrt(12 * L / (m * mu)) - 4 and m < 3 * L / (4 * mu)):
        thetas = np.ones(T) * gamma * (alpha + p) / alpha
        thetas[-1] = gamma / alpha
    else:
        thetas = np.zeros(T)
        Gamm_t = 1
        for t in range(1, T):
            thetas[t-1] = Gamm_t * (1 - (1 - alpha - p) * (1 + mu * gamma))
            Gamm_t *= (1 + mu * gamma)
        thetas[-1] = Gamm_t
    return thetas


def varag(y_0, oracle, oracle_full, mu, L, S, R):
    """Use Varag to minimize g(y)."""
    m = len(distr)
    s_0 = get_s_0(m)

    y = y_0.copy()
    y_tilde = y_0.copy()
    for s in range(S):
        all_grads = oracle_full(y_tilde)
        grad_full = all_grads.mean(axis=1)
        y_t = y.copy()
        y_t_bar = y_tilde.copy()
        T, gamma, alpha, p = get_params(s, s_0, L, mu, m)
        y_bars = np.zeros((y_t_bar.shape[0], T))
        for t in range(T):
            i = randrange(m)
            y_underbar = get_y_underbar(mu, gamma, alpha, p, y_t_bar, y_t, y_tilde)
            grad = oracle(y_underbar, i)
            G = get_G(grad_full, all_grads[:, i], grad)
            y_t = argmin_y(gamma, G, mu, y_underbar, y_t, R)
            y_t_bar = (1 - alpha - p) * y_t_bar + alpha * y_t + p * y_tilde  # get_y_t_bar(y_t_bar, alpha, p, y_t, y_tilde)
            y_bars[:, t] = y_t_bar
        y = y_t
        thetas = get_thetas(gamma, alpha, p, T, L, mu, s, s_0, m)
        y_tilde = y_bars @ thetas / thetas.sum()  # get_y_tilde(y_bar_list, thetas)
    return y_tilde


# def erm_experiment():
#     data = load_svmlight_file("covtype.libsvm.binary.scale")
#     m = 10000
#     Z, t = data[0][:m], data[1][:m]
#     m, n = Z.shape
#     Z = np.hstack((Z.todense(), np.ones((m, 1))))
#     t = np.array(t * 2 - 3)
#
#     y_0 = np.random.randn((n, 1))
#
#     R = 100
#     S = 10
#
#     def oracle_full(y):
#         return np.log((x + 45) * np.exp(-1) / 50) + 1
#
#     x_opt, vals = varag(y_0, oracle, oracle_full, mu, L, S, R)
#     print(x_opt)
#     plt.plot(vals)
#     plt.savefig(f"plots/test_{datetime.datetime.now().time()}.png")


if __name__=="__main__":
    entropy_experiment()
