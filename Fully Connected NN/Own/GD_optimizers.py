import numpy as np
def sgd(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw
    return w, config

def rmsprop(x, dx, config=None):
    if config is None: config = {}
    learning_rate = config.setdefault('learning_rate', 1e-2)
    decay_rate = config.setdefault('decay_rate', 0.99)
    eps = config.setdefault('epsilon', 1e-8)
    cache = config.setdefault('cache', np.zeros_like(x))
    cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
    next_x = x - learning_rate * dx / (np.sqrt(cache) + eps)
    config['cache'] = cache
    return next_x, config


def adam(x, dx, config=None):
    if config is None: config = {}
    learning_rate = config.setdefault('learning_rate', 1e-3)
    beta1 = config.setdefault('beta1', 0.9)
    beta2 = config.setdefault('beta2', 0.999)
    eps = config.setdefault('epsilon', 1e-8)
    m = config.setdefault('m', np.zeros_like(x))
    v = config.setdefault('v', np.zeros_like(x))
    t = config.setdefault('t', 1)
    t += 1
    m = beta1 * m + (1 - beta1) * dx
    mt = m / (1 - beta1 ** t)
    v = beta2 * v + (1 - beta2) * (dx ** 2)
    vt = v / (1 - beta2 ** t)
    next_x = x - learning_rate * mt / (np.sqrt(vt) + eps)
    config['t'] = t
    config['m'] = m
    config['v'] = v
    return next_x, config
