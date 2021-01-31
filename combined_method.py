import numpy as np
from vaidya import vaidya, get_init_polytope
from varag import varag
from operator import itemgetter


def combined_method(x_0, y_0, dF_dx, dFi_dy, dF_dy_full, vaidya_params, varag_params):
    d, Rx, eps, K, newton_steps = itemgetter('d', 'Rx', 'eps', 'K', 'newton_steps')(vaidya_params)
    n, Ry, m, S, mu, L = itemgetter('n', 'Ry', 'm', 'S', 'mu', 'L')(varag_params)

    def get_y_opt(x, y_0):
        oracle_y = lambda y, i: dFi_dy(x, y, i)
        oracle_y_full = lambda y: dF_dy_full(x, y)
        pts, grad_evals = varag(y_0, m, oracle_y, oracle_y_full, mu, L, S, Ry)
        return pts[-1], grad_evals[-1]

    def oracle_x(x, y_0):
        y_opt, n_grad_evals = get_y_opt(x, y_0)
        return dF_dx(x, y_opt), y_opt, n_grad_evals

    A_0, b_0 = get_init_polytope(d, Rx)

    xs, ys, aux_evals = vaidya(A_0, b_0, x_0, y_0, eps, K, oracle_x, newton_steps=newton_steps)
    return xs, ys, aux_evals
