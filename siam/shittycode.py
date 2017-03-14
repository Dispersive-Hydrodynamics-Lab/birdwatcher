#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.stats import norm
from scipy.stats.mstats import mquantiles



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

x = np.linspace(0, 1, 10000)
def gen_exp(l):
    return lambda x: l * np.exp(-l * x)
plt.figure()
plt.plot(x, gen_exp(2)(x), 'r-', label=r'$\lambda = 1$')
plt.plot(x, gen_exp(5)(x), 'b-', label=r'$\lambda = 5$')
plt.plot(x, gen_exp(10)(x), 'g-', label=r'$\lambda = 10$')
plt.legend(loc=0)
plt.savefig('./siam/img/exponential.png')

x = np.linspace(-3, 3, 1000)
x1 = np.linspace(0, .4, 10)
z = np.ones(len(x1))
fig, axarr = plt.subplots(2, 1)
axarr[0].plot(x, norm.pdf(x))
axarr[0].fill_between(x, norm.pdf(x), 0, alpha=0.25)
axarr[0].plot(-0.67 * z, x1, 'r-')
axarr[0].plot(0 * z, x1, 'r-')
axarr[0].plot(0.67 * z, x1, 'r-')
axarr[0].set_title('Quartiles')
axarr[0].set_ylim(0, 0.4)
axarr[0].axis('off')
axarr[1].plot(x, norm.pdf(x))
axarr[1].fill_between(x, norm.pdf(x), 0, alpha=0.25)
axarr[1].plot(-1.28 * z, x1, 'r-')
axarr[1].plot(-0.84 * z, x1, 'r-')
axarr[1].plot(-0.52 * z, x1, 'r-')
axarr[1].plot(-0.25 * z, x1, 'r-')
axarr[1].plot(0 * z, x1, 'r-')
axarr[1].plot(0.25 * z, x1, 'r-')
axarr[1].plot(.52 * z, x1, 'r-')
axarr[1].plot(.84 * z, x1, 'r-')
axarr[1].plot(1.28 * z, x1, 'r-')
axarr[1].set_title('10-tiles')
axarr[1].set_ylim(0, 0.4)
axarr[1].axis('off')
plt.tight_layout()
plt.savefig('./siam/img/quantiles.png')

plt.figure()
data = np.random.normal(size=60)
real = np.random.normal(size=10000)
x = np.linspace(0, 1, 30)
x1 = np.linspace(-3, 3, 10)
theoretical_quantiles = mquantiles(data, prob=x)
real_quantiles = mquantiles(real, prob=x)
plt.plot(x1, x1, 'k-')
for p0, p1 in zip(theoretical_quantiles, real_quantiles):
    xpoints = (p0, p0)
    ypoints = (p0, p1)
    plt.plot(xpoints, ypoints, 'r-')
plt.plot(theoretical_quantiles, real_quantiles, 'ko')
plt.savefig('./siam/img/qqplot.png')
