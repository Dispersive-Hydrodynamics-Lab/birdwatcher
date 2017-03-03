#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial



x = np.arange(21)
def gen_pois(l):
    return lambda x: (l**x * np.exp(-l)) / factorial(x)
plt.figure()
plt.plot(x, gen_pois(.5)(x), 'mo-', label=r'$\lambda = 0.5$')
plt.plot(x, gen_pois(1)(x), 'r^-', label=r'$\lambda = 1$')
plt.plot(x, gen_pois(5)(x), 'bd-', label=r'$\lambda = 5$')
plt.plot(x, gen_pois(10)(x), 'gs-', label=r'$\lambda = 10$')
plt.legend(loc=0)
plt.savefig('./siam/img/poisson.png')
