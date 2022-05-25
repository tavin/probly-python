import jax
from jax import jacfwd, jacrev
from jax import jit
from jax import numpy as np
from jax.lax import while_loop
from jax.scipy.special import gammaln


GRID_SIZE = 2**15
OPTIM_RATE = 5e-2
MAX_ITER = 1_000


def locate(k, d):

    # well, eventually this is going away... https://github.com/google/jax/issues/8178
    assert jax.config.read('jax_enable_x64')

    q = np.linspace(0, 1, num=GRID_SIZE)[1:-1]  # 0 < q < 1
    log_Ck = np.log(0.5)*k/2 - gammaln(k/2)

    def loss(theta):
        """ KL divergence """
        log_alpha, log_beta = theta[0:2]
        alpha, beta = np.exp(log_alpha), np.exp(log_beta)
        gamma = np.array([-theta[2:].sum()-1, *theta[2:], 0.])
        t = np.polyval(gamma, q)
        dt_dq = np.polyval(np.polyder(gamma), q)
        log1p_t = np.log1p(t)
        log_pdf = log_Ck + (k/2-1) * (beta * np.log(-log1p_t) - log_alpha) - 0.5 * np.power(-log1p_t, beta) / alpha
        log_der = log_alpha - log_beta - np.expm1(log_beta) * np.log(-log1p_t) + log1p_t - np.log(-dt_dq)
        return (log_pdf - log_der).mean()

    gradient = jacrev(loss)
    hessian = jacfwd(gradient)
    step = jit(lambda theta: theta - OPTIM_RATE * np.linalg.pinv(hessian(theta)) @ gradient(theta))

    def cond(state):
        it, done, theta = state
        return (it < MAX_ITER) & ~done

    def loop(state):
        it, done, theta = state
        new_theta = step(theta)
        done = np.allclose(theta, new_theta, equal_nan=True, rtol=1e-6, atol=1e-9)
        return it+1, done, new_theta

    # it's very sensitive to these initial parameters
    theta = np.array([-np.log(k), -0.5*np.log(k/2-4/3)] + [0.]*(d-2) + [-1.])
    it, done, theta = while_loop(cond, loop, (0, False, theta))

    assert done, 'did not converge'
    assert np.isfinite(theta).all(), 'theta blew up'

    return theta


def ICDF(k, d, dtype=None):

    theta = locate(k, d)
    log_alpha, log_beta = theta[0:2]
    alpha, beta = np.exp(log_alpha), np.exp(log_beta)
    gamma = np.array([-theta[2:].sum()-1, *theta[2:], 0.])

    if dtype:
        alpha, beta, gamma = (var.astype(dtype) for var in (alpha, beta, gamma))

    def icdf(q):
        t = np.polyval(gamma, q)
        x = np.power(-np.log1p(t), beta) / alpha
        return x

    return icdf
