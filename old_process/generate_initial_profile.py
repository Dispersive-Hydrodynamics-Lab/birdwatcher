#!/usr/bin/env python3

import sys
import os
import argparse
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mplt
import scipy.io as sco
import scipy.interpolate as sci
import scipy.misc as scm
import scipy.ndimage.filters as scfilt
import scipy.signal as scsig
import scipy.stats as sc_st
from tqdm import tqdm

from typing import Dict, Tuple, List, Optional, TypeVar
from enforce import runtime_validation


MAGIC_NUMBER = 20

H = 0.01


def main():
    create_solitons('output.mat')

def create_solitons(filename: str) -> None:
    """
    Create initial soliton profile for solver. Saves output as .mat file
    """
    domain = np.arange(0, 500, H)
    # number_of_solitons = np.random.randint(30, 60)
    number_of_solitons = 16
    soliton_heights = get_heights(number_of_solitons, vers=1)
    # soliton_positions = 20 * np.random.choice(np.arange(5, 95), size=number_of_solitons, replace=False)
    soliton_positions = np.linspace(25, 475, number_of_solitons)
    profile = np.ones(len(domain))
    print('Generating {}'.format(number_of_solitons))
    for position, height in zip(soliton_positions, soliton_heights):
        print('\tSoliton @ {:0.01f} with h={:0.01f}'.format(position, height))
        soliton = get_soliton(domain, height, position)
        profile += soliton
    domain = np.arange(0, 1000, H)
    new_profile = np.ones(len(domain))
    new_profile[1000:len(profile) + 1000] = profile
    profile = new_profile


    derivatives = get_derivative(profile)

    fig, axarr = plt.subplots(2, 1, figsize=(20, 20))
    axarr[0].plot(domain, profile, 'r-')
    axarr[0].set_title('Initial Profile')
    axarr[1].plot(domain, derivatives, 'b-')
    axarr[1].set_title('1st Derivative')
    axarr[0].set_ylim(1, 6)
    axarr[0].get_xaxis().set_major_locator(mplt.MultipleLocator(50))
    axarr[0].get_xaxis().set_major_formatter(mplt.FormatStrFormatter('%d'))
    axarr[0].get_xaxis().set_minor_locator(mplt.MultipleLocator(10))
    plt.savefig('./out.png')
    plt.show()

    output = {'domain': domain, 'area': profile, 'area_derivative': derivatives}
    sco.savemat(filename, output)

def get_heights(count, vers=1):
    heights = None
    pre_comp = np.arange(2.0, 7.5, 0.5)
    if vers == 1:
        heights = np.random.choice(np.arange(1, 5.1, 0.5), size=count, replace=True)
    # elif vers == 2:
    #     rv = sc_st.arcsine(scale=10)
    #     heights = rv.rvs(count)
    #     bins, edges = np.histogram(heights, bins=pre_comp)
    #     heights = []
    # sys.exit(0)
    return heights

def get_soliton(domain: np.ndarray, amplitude: int, position: int) -> np.ndarray:
    """
    Gaussian Approximation to soliton.
    """
    soliton = sco.loadmat('./solitons/conduit{:g}.mat'.format(amplitude))
    interpolation = sci.InterpolatedUnivariateSpline(soliton['z'], soliton['phi'])
    if np.abs(interpolation(domain - position) > 20).any():
        return get_soliton(domain, amplitude + 1, position)
    return interpolation(domain - position)


def get_derivative(f: np.ndarray) -> np.ndarray:
    """
    Derivative of Gaussian Approximation

    https://en.wikipedia.org/wiki/Finite_difference_coefficient

    Let's use central finite difference with O(h^8) accuracy

    Assume that the beginning and ends of the arrays are zero, and that the first derivative is zero.
    """
    MAGIC_NUMBERS = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
    derivative = np.zeros(len(f))
    for i in range(4, len(f) - 5):
        for num_index, f_index in zip(range(9), range(-4, 6)):
            derivative[i] += MAGIC_NUMBERS[num_index] * f[i + f_index]
        derivative[i] /= H
    # IM PARANOID
    derivative[:20] = 0
    derivative[-20:] = 0
    return derivative


if __name__ == '__main__':
    sys.exit(main())
