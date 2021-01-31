import numpy as np
from random import randrange, seed


def get_s_0(m):
    return int(np.log2(m)) + 1


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
    return int(T), gamma, alpha, p


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
    if mu == 0 or s <= s_0 or (s_0 < s <= s_0 + np.sqrt(12 * L / (m * mu)) - 4 and m < 3 * L / (4 * mu)):
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


def varag(y_0, m, oracle, oracle_full, mu, L, S, R):
    """Use Varag to minimize g(y) over Euclidean ball of radius R."""
    s_0 = get_s_0(m)
    pts = [y_0.copy()]
    grad_evals = [0]

    y = y_0.copy()
    y_tilde = y_0.copy()
    for s in range(1, S+1):
        all_grads = oracle_full(y_tilde)
        grad_full = all_grads.mean(axis=1, keepdims=True)
        y_t = y.copy()
        y_t_bar = y_tilde.copy()
        T, gamma, alpha, p = get_params(s, s_0, L, mu, m)
        grad_evals.append(grad_evals[-1] + m + T)
        y_bars = np.zeros((y_t_bar.shape[0], T))
        for t in range(1, T+1):
            seed(s * 2 * m + t - 1)
            i = randrange(m)
            y_underbar = get_y_underbar(mu, gamma, alpha, p, y_t_bar, y_t, y_tilde)
            grad = oracle(y_underbar, i)
            G = get_G(grad_full, all_grads[:, i:i+1], grad)
            y_t = argmin_y(gamma, G, mu, y_underbar, y_t, R)
            y_t_bar = (1 - alpha - p) * y_t_bar + alpha * y_t + p * y_tilde
            y_bars[:, t-1] = np.squeeze(y_t_bar)
        y = y_t
        thetas = get_thetas(gamma, alpha, p, T, L, mu, s, s_0, m)
        y_tilde = y_bars @ thetas[:, None] / thetas.sum()
        pts.append(y_tilde)
    return pts, grad_evals
