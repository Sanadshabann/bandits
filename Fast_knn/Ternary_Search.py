import numpy as np
import math as m


def uncertainty(k, actions, k_nearest_sorted, theta, phi, a):
    t = np.shape(actions)[0]
    k_least = k_nearest_sorted[:k, 0].astype(int)
    N = sum(actions[k_least] == a)
    U = m.sqrt(theta * m.log(t) / N) + phi(t) * k_nearest_sorted[k - 1, 1] if (N != 0) else m.inf
    return U


def k_ternary_search(actions, k_nearest_sorted, theta, phi, a, k_max):
    f = lambda k: uncertainty(k, actions, k_nearest_sorted, theta, phi, a)
    l = 1
    r = k_max - 1
    while True:
        c1 = l + m.floor((r - l) / 3)
        c2 = l + 2 * m.floor((r - l) / 3)
        u1 = f(c1)
        u2 = f(c2)
        if u1 > u2:
            l = c1
        elif u1 < u2:
            r = c2
        else:
            if c1 == c2:
                return c1, u1
            else:
                l, r = c1, c2
        if abs(l - r) <= 1:
            break
    return m.floor((l + r) / 2), f(m.floor((l + r) / 2))

