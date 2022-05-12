import numpy as np
from scipy import stats


def polynomial_transform(k, d):

    """
    For X ~ Chi2(k), let q = cdf(x) and t = exp(- (x/k) ^ sqrt(k/2)).
    Solve for a in the power series $t = 1 + a_1 q + a_2 q^2 + \dots$.

    :param k: chi2 parameter
    :param d: polynomial degree
    :return: ndarray vector of coefficients [a_d, ..., a_1, 1.0]
    """

    # even if outer np is jax.numpy, this routine should use plain (64-bit) numpy
    import numpy as np

    z = np.finfo(np.array([]).dtype).tiny
    t = np.logspace(0, np.log10(z), num=2**15)

    x = k * np.power(-np.log(t), 1./np.sqrt(k/2.))
    q = stats.chi2.cdf(x, k)
    t, q = np.hstack([t, 0.]), np.hstack([q, 1.])
    M = np.power(q[:, None], range(d+1))  # M_i = 1 + a_1 q + a_2 q^2 + ... for each q_i

    # a_0 must be 1 so drop that equation; else lstsq finds a_0 ≈ 1.0, but less
    M = M[1:, 1:]
    t = t[1:] - 1.0

    # solution by least squares fit is far more accurate than "exact" solution
    a, *_ = np.linalg.lstsq(M, t, rcond=None)
    a = np.hstack([1.0, a])

    # numpy polynomials put leading order first
    return np.flip(a)


class ICDF:

    def __new__(cls, k, d):

        """
        Return a (vectorized) function icdf(q) that approximates x, where X ~ Chi2(k) and q = cdf(x).

        :param k: chi2 parameter
        :param d: polynomial degree
        """

        dtype = np.array([]).dtype
        tiny = np.finfo(dtype).tiny
        poly = polynomial_transform(k, d).astype(dtype)

        def icdf(q):
            t = np.polyval(poly, q)
            t = np.clip(t, tiny, None)
            return k * np.power(-np.log(t), 1./np.sqrt(k/2.))

        return icdf
