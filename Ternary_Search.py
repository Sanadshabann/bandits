import numpy
import math as m


def uncertainty(k, actions, distances, theta, phi, a):
    t = len(actions)
    k_least = numpy.argpartition(distances, k - 1)[:k]
    N = sum(numpy.array(actions)[k_least] == a)
    uncer = m.sqrt(theta * m.log(t) / N) + phi(t) * distances[k_least[-1]] if (N != 0) else m.inf
    return uncer


def k_ternary_search(actions, distances, theta, phi, a):
    f = lambda k: uncertainty(k, actions, distances, theta, phi, a)
    l = 0
    r = min(len(actions) - 1, 200)
    while True:
        c1 = l + m.floor((r - l) / 3)
        c2 = l + 2 * m.floor((r - l) / 3)
        diff = f(c1) - f(c2)
        if diff > 0:
            l = c1
        elif diff < 0:
            r = c2
        else:
            if c1 == c2:
                return c1
            else:
                l, r = c1, c2
        if abs(l - r) <= 1:
            break
    return m.floor((l + r) / 2)
