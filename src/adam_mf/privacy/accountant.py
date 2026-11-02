"""GDP calibration for a whole fixed-epoch matrix mechanism, without sampling."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr


def gdp_delta(mu, epsilon):
    # Stable evaluation of Phi(a) - exp(epsilon) Phi(b).
    a = -epsilon / mu + mu / 2
    b = -epsilon / mu - mu / 2
    log_a = log_ndtr(a)
    return float(np.exp(log_a) * (-np.expm1(epsilon + log_ndtr(b) - log_a)))


def target_mu(epsilon, delta):
    return brentq(lambda mu: gdp_delta(mu, epsilon) - delta, 1e-6, 100.0)


def epsilon_from_mu(mu, delta):
    if mu == 0 or gdp_delta(mu, 0) <= delta:
        return 0.0
    return brentq(lambda eps: gdp_delta(mu, eps) - delta, 0.0, 1000.0)


def fixed_epoch_sensitivity(strategy, k, b_participation, steps=None):
    """L2 bound over patterns j, j+b, ..., j+(k-1)b.

    Uses the absolute Gram bound (Eq. 5 of BandMF), allowing different
    bounded gradient directions on each participation. For our nonnegative
    strategies it is exact. Adjacency is add/remove with a zeroed contribution.
    A prefix retains the original full-trajectory participation schedule.
    """
    strategy = np.asarray(strategy, dtype=np.float64)
    assert strategy.shape == (k * b_participation, k * b_participation)
    rows = strategy if steps is None else strategy[:steps]
    gram = rows.T @ rows
    squared = [np.abs(gram[np.ix_(idx, idx)]).sum()
               for j in range(b_participation)
               for idx in [j + b_participation * np.arange(k)]]
    return float(np.sqrt(max(squared)))


def calibrate(strategy, config):
    p = config['privacy']
    mu = target_mu(p['epsilon'], p['delta'])
    sensitivity = fixed_epoch_sensitivity(strategy, p['k'], p['b_participation'])
    return {'target_mu': mu, 'sensitivity': sensitivity,
            'innovation_std_sum': p['max_grad_norm'] * sensitivity / mu,
            'adjacency': p['adjacency'], 'k': p['k'],
            'b_participation': p['b_participation'],
            'sampling_amplification': False}
