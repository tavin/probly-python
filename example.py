import jax
from scipy import stats

import chi2


if __name__ == '__main__':
    k, d = 3, 11
    chi2.np = jax.numpy
    print(chi2.polynomial_transform(k, d))
    icdf = chi2.ICDF(k, d)
    for q in jax.numpy.linspace(0, 1, num=5):
        x = icdf(q)
        print(q, x, stats.chi2.cdf(x, k))
