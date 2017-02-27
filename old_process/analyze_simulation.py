#!/usr/bin/env python3.5

# stdlib
import sys
import os
import argparse
import concurrent.futures
import random
import math

# third party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn
import moviepy
from moviepy import editor
import scipy.io as sco
import scipy.misc as scm
import scipy.ndimage.filters as scfilt
import scipy.signal as scsig
import scipy.stats as sc_st
from tqdm import tqdm

from typing import Dict, Tuple, List, Optional, TypeVar, Dict
from enforce import runtime_validation

# local
import util


NUMLINES = 100


def main():
    args = get_args()
    print(args)

    if args.directory:
        print("DIRECTORY")
        print('Processing {}'.format(args.directory))
        print('=' * (len(args.directory) + 11))
        output_filename = os.path.split(args.directory)[-1]
        if args.show:
            show_numerics(args.directory, output_filename, args.write)
        data = convert_dir_to_file(args.directory, output_filename, args.write)

    if args.file:
        print("FILE")
        print('Processing {}'.format(args.file))
        print('=' * (len(args.file) + 11))
        output_filename = os.path.split(args.file)[-1].split('.')[0]
        data = sco.loadmat(args.file)['data']

    print('Plotting Phase Portrait')
    phase_portrait(data, args.write, args.show, output_filename)

    print('Plotting Solitons')
    peaks, solitons, soliton_hash = plot_solitons(data, output_filename,
                                                  show=args.show,
                                                  write=args.write)

    continuous_rv = to_continuous_rv(solitons, soliton_hash, data)
    continuous_qq = get_continuous_qq(continuous_rv)


def get_continuous_qq(data: Dict[float, np.ndarray]):
    """
    We want to compare our data vs continuous poisson process on a QQ plot.
    """
    prob_val_num = int(np.mean([len(tup[1]) for tup in data.items() if len(tup[1]) not in [0, 1, 2]]))
    errors = []
    psize = []
    x = []
    for pos, times in sorted(data.items(), key=lambda tup: tup[0]):
        if len(times) in [0, 1, 2]:
            continue
        x.append(pos)
        # first determine t - the length of our interested set
        points = times - times.min()
        t = points.max()
        # now measure λ from dataset - average points per unit length.
        λ = np.mean([1 if i in points else 0 for i in range(t)])
        # now simulate a HPP with same intensity (using code from markov class)
        T, N = HPP(λ, t)
        # Calculate each of their quantiles
        prob_vals = np.linspace(0, 1, prob_val_num)
        points_quantiles = sc_st.mstats.mquantiles(points, prob=prob_vals)
        T_quantiles = sc_st.mstats.mquantiles(T, prob=prob_vals)

        # we can now plot QQ if want
        # plt.figure()
        # plt.scatter(points_quantiles, T_quantiles)
        # plt.show()

        residuals = np.sum((points_quantiles - T_quantiles)**2)
        psize.append(len(times))
        errors.append(residuals)

    x = np.array(x)
    coeff = np.polyfit(x, errors, 1)
    plt.figure(figsize=(8, 8))
    plt.scatter(x, errors, s=psize, alpha=0.6)
    plt.plot(x, coeff[0] * x + coeff[1])
    plt.xlabel('Tracking Line Position')
    plt.ylabel('Residual Sum of Squares Error')
    plt.title('Q-Q Plot Error')
    plt.savefig('qq_continuous')


def HPP(l, t):
    T = [0]
    i = 0
    while True:
        u = random.random()
        i += 1
        T.append(T[i - 1] - math.log(u) / l)
        if T[i] > t or i > 10000:
            N = i - 1
            break
    return T, N



def to_continuous_rv(solitons: Dict[str, Tuple[int, np.ndarray]],
                     soliton_hash: Dict[str, Dict[int, np.ndarray]],
                     data: np.ndarray) -> Dict[float, np.ndarray]:
    """
    So we can convert this to a continuous poisson process if we draw a
    line in "space" and then measure the intensity over "time"

    The data array is the raw output of the simulation, where
        height = space
        width = time

    """
    # We demarcate lines in "SPACE"
    pos_data = {}
    plt.figure(figsize=(8, 8))
    for pos in np.linspace(1, len(data), NUMLINES):
        pos_data[pos] = []
        # Now for each soliton we need to determine when it crosses this line
        for soliton_name, path in solitons.items():
            # convert `path` to tuple of (time, zpos)
            for i in range(5, len(path)):
                time = path[i][0],
                zpos = path[i][1][0]
                old_zpos = path[i - 1][1][0]
                if zpos > pos and old_zpos < pos:
                    pos_data[pos].append(time[0])
                    break
        plt.scatter(pos * np.ones(len(pos_data[pos])), pos_data[pos])
        pos_data[pos] = np.array(sorted(pos_data[pos]))
    plt.savefig('cont_rv.png')
    return pos_data


def convert_dir_to_file(data_dir: str, output_filename: str, write: bool):
    parameters = sco.loadmat(os.path.join(data_dir, 'parameters.mat'))
    files = [os.path.join(data_dir, _)
             for _ in sorted(os.listdir(data_dir))][1:-1]

    data = {'zmax': parameters['zmax'], 't': parameters['t']}
    data_arr = np.zeros((len(sco.loadmat(files[0])['A']), len(files)))
    for i, file in enumerate(files):
        file_data = sco.loadmat(file)
        data_arr[:, i] = file_data['A'][:, 0]
    data['data'] = data_arr

    if write:
        sco.savemat('{}.mat'.format(output_filename), data)
    return data


def show_numerics(data_dir: str, output_filename: str, write: bool):
    """
    Show numerical output from solver
    """
    def read_data(filename):
        data = sco.loadmat(filename)['A']
        return data
    files = [os.path.join(data_dir, _)
             for _ in sorted(os.listdir(data_dir))][1:-1]
    parameters = sco.loadmat(os.path.join(data_dir, 'parameters.mat'))
    zmax = parameters['zmax'][0][0]
    tvals = parameters['t']
    data_length = len(read_data(files[0]))

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    line, = ax.plot(np.linspace(0, zmax, data_length),
                    read_data(files[0]))
    ax.set_ylim(0, 6.5)

    def update(i, l):
        if i % 10 == 0:
            print(i, end=' ', flush=True)
        l.set_data(np.linspace(0, zmax, data_length),
                   read_data(files[i]))
        return l,

    interval = 10
    ani = animation.FuncAnimation(fig, update, len(files),
                                  fargs=(line,),
                                  interval=interval, blit=True)
    if write:
        ani.save('{}.mp4'.format(output_filename))


def phase_portrait(data, write, show, filename):
    """
    Plots phase portrait for current dataset

    :param data: np.ndarray => 2d array of values
    :param write: bool => Whether or not to save the plot
    :param show: bool => whether or not to show the plot
    :param filename: str => the savename of the file

    :return: None
    """
    if write:
        plt.imsave('./output/{}_phaseportrait.png'.format(filename),
                   scm.imresize(data, (1000, 1000)))

    if show:
        plt.figure()
        plt.imshow(scm.imresize(data, (1000, 1000)))
        plt.show()

def plot_solitons(data: np.ndarray,
                  filename: str,
                  show: Optional[bool]=False,
                  write: Optional[bool]=False):
    """
    Plot (and animate) the solitons.

    :param data: 2d array of values
    :param filename: the savename of the file
    :param show: whether or not to show the plot
    :param write: Whether or not to save the plot

    :return peaks: contains a 2d np.ndarray for every frame
    :return solitons: soliton name to array of times and positions
    :return soliton_hash: soliton name to dictionary of times to positions

    TODO: need to auto config parameters:
        * gaussian filter parameter
        * threshold
        * find_peaks parameter
        * x and y limits
        * Frame intervals
    """
    # Initialize datapoints
    domain = np.arange(len(data))
    cdata = data[:, 0]   # First column

    # Smooth the data for more accuracy, only need for raw data
    print('\tSmoothing Data')
    sdata = scfilt.gaussian_filter1d(data, 0, axis=0)
    scdata = sdata[:, 0]

    # Establish figure
    print('\tInitializing Figure')
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Plot Raw Data
    line1, = ax.plot(domain, cdata, 'b-', alpha=0.5)
    # Plot Smoothed Data
    line, = ax.plot(domain, scdata, 'b-')

    vert_domain = np.arange(0, 10)
    ax.plot(len(data) * 0.25 * np.ones(len(vert_domain)), vert_domain, 'r-')

    # Set figure axes limits
    ax.set_ylim(0, data.max() + 0.1 * data.max())
    ax.set_xlim(0, domain.max())

    # Grab and plot peaks
    if input('\tgenerate new peaks? (y/n) ') == 'y':
        peaks = get_peaks(domain, sdata)
        np.save('{}.npy'.format(filename), peaks)
    else:
        peaks = np.load('{}.npy'.format(filename))

    # Identify Solitons
    solitons, soliton_hash = get_solitons(peaks)

    # Plot Initial Peaks
    points, = ax.plot(peaks[0][:, 0], peaks[0][:, 1],
                      linestyle='', marker='o', color='r')

    # Annotate Initial Solitons (set to (0, 0) if not present)
    annotations = []
    labels = list(solitons.keys())
    default_pos = (0, 0)
    for key in labels:
        value = soliton_hash[key]
        pos = default_pos
        if 0 in list(value.keys()):
            pos = (value[0][0], value[0][1])
        an = ax.annotate(r'${}$'.format(key),
                         xy=pos, size=20)
        annotations.append(an)

    # Cover up hidden annotations with little white box. Currently not working.
    rectangle = plt.Rectangle((0, 0), 250, 20, fc='w', ec='k', clip_on=True)
    ax.add_patch(rectangle)

    def animate(i: int):
        """ Animation function for plot """
        if i % 100 == 0:
            print(i, sep=' ')
        cdata = data[:, i]  # New datapoints
        scdata = sdata[:, i]  # New Smoothed datapoints
        line.set_ydata(scdata)
        line1.set_ydata(cdata)
        points.set_data(peaks[i][:, 0], peaks[i][:, 1])

        ax.set_title('t=kajsdfh')

        for j in range(len(labels)):
            key = labels[j]
            positions = soliton_hash[key]
            annotation = annotations[j]
            pos = default_pos
            if i in list(positions.keys()):
                pos = (positions[i][0], positions[i][1])
            annotation.set_position(pos)
        return line, annotations

    def init():
        """ Renders the first frame """
        return line, annotations

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(data[0])),
                                  init_func=init,
                                  interval=50)

    if write:
        print('Animating')
        ani.save('{}.mp4'.format(filename))
        # editor.VideoFileClip('anim.mp4')\
        #         .write_gif('anim.gif')
    if show:
        print('Animating')
        plt.show()

    return peaks, solitons, soliton_hash


def peaks_func(d, domain):
    """
    Determine peaks for frame

    :param d: np.ndarray => 1d array of y values
    :param domain: np.ndarray => 1d array of x values

    :return: np.ndarray => array of XY coordinates of the positions
    """
    thresh = 1.1  # TODO: Make this adjustable - issue#2
    peakrange = np.arange(1, 100)
    rawpeaks = np.array(scsig.find_peaks_cwt(d, peakrange))
    rawpeaks = rawpeaks[rawpeaks > rawpeaks.min()]  # Remove leftmost peak (false positive at input)
    threshpeaks = rawpeaks[d[rawpeaks] > thresh]
    return np.array([domain[threshpeaks], d[threshpeaks]]).T


def get_peaks(domain, sdata):
    """
    Parallelize PeakFinding

    https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example
    Obviously using ProcessPools instead

    :param domain: np.ndarray => 1d array of x values
    :param sdata: np.ndarray => 2d array of smoothed y values

    :return: list => sorted list of peaks by frame
    """
    peaks = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(peaks_func, sdata[:, i], domain): i for i in
                   range(len(sdata[0]))}
        print('Number of workers: {}'.format(len(futures)))
        for future in tqdm(concurrent.futures.as_completed(futures),
                           desc='Calculating Peaks', leave=True, total=len(sdata[0])):
            i = futures[future]
            peaks.append((i, future.result()))
    return [x[1] for x in sorted(peaks, key=lambda tup: tup[0])]


def get_solitons(peaks):
    """
    Isolate and track peak transitions
    General idea here is to see what points are closest to old points
    Need to also track when things disappear off the edge....
    Ideally we get a 3dimensional array of soli,x,y
    TODO: When edge is closer than next nearest point, assume it fell off and adjust /EVERY LABEL/.

    :param peaks: list => contains a 2d np.ndarray for every frame

    :return solitons: dictionary => soliton name to array of times and positions
    :return soliton_hash: dictionary => soliton name to dictionary of times to positions
    """

    def names():
        """ Generator for Unicode soliton names """
        i = 913
        while True:
            yield chr(i)
            i += 1

    name = names()
    solitons = {}
    soliton_hash = {}
    for i in tqdm(range(len(peaks) - 1), desc='Calculating Soliton Transitions', leave=True):
        cstate = peaks[i]
        nstate = peaks[i + 1]

        for csoli in cstate:  # For each point in the current state
            distances = np.zeros(len(nstate))
            # Check distance between cpoint and (potential) npoint
            for j in range(len(nstate)):
                distances[j] = np.linalg.norm(csoli - nstate[j])
            nsoli = nstate[np.argmin(distances)]  # The next state is min distance away
            # At this point we have the location of the current soliton position as well as its next
            # position. We need to search the soliton index to find an instance of the current
            # position (if it exists). If it doesn't exist, add it
            for key, value in solitons.items():
                time = value[-1][0]
                last_position = value[-1][1]
                if np.array_equal(last_position, csoli):
                    solitons[key] += [(i, nsoli)]
                    soliton_hash[key][i] = nsoli
                    break
            else:
                nkey = next(name)
                solitons[nkey] = [(i, nsoli)]
                soliton_hash[nkey] = {i: nsoli}
    return solitons, soliton_hash

def solitons_to_rv(solitons, soliton_hash, maxtime, maxpos, binsize=100):
    """
    Converts the solitons that have been found to a random variable.

    :param solitons: dictionary => soliton name to array of times and positions
    :param soliton_hash: dictionary => soliton name to dictionary of times to z positions
    :param maxtime: int => the max "time" (width of phase portrait)
    :param maxpos: int => the max "position" (height of phase portrait)
    :param binsize: (int) => how many seconds to elapse before recording value

    :returns: np.ndarray => 1d array of values
    """

    # First get max and min times for solitons
    soliton_info = {}
    for soliton, indices in soliton_hash.items():
        times = sorted(t for t, p in indices.items())
        soliton_info[soliton] = (min(times), max(times))

    positions = {int((1.0 / NUMLINES) * maxpos * x):[] for x in range(NUMLINES)}
    for key in positions:
        crossed = 0
        for i in range(0, maxtime, binsize):
            count = 0
            for soliton, indices in soliton_hash.items():
                try:
                    cpos = indices[i][0]
                except KeyError:
                    if i < soliton_info[soliton][0]:
                        cpos = -1
                    else:
                        cpos = maxpos
                if cpos >= key:  # if the current position is greater than our line
                    count += 1
            positions[key].append(count - crossed)
            crossed = count
    # strip leading zeros
    shortest = 10000
    for key in positions:
        for i, item in enumerate(positions[key]):
            if item != 0:
                break
        positions[key] = positions[key][i:]
        if len(positions[key]) < shortest:
            shortest = len(positions[key])
    # Make each rv the same size.
    for key in positions:
       if len(positions[key]) > shortest:
           positions[key] = positions[key][-shortest:]
    return positions


def get_empirical_cdf(data, k_vals):
    """
    Finds the empirical cdf from a poisson dataset.

    :param data: np.ndarray => data for analysis
    :param k_vals: np.ndarray => integers to use

    :return: np.ndarray for cdf
    """
    cdf = np.zeros(len(k_vals))
    n = len(data)
    for i in range(len(k_vals)):
        cdf[i] = len(data[data <= k_vals[i]])
    cdf *= (1 / n)
    return cdf


def get_poisson_cdf(l, k_vals):
    """
    Finds the "true" poisson cdf for given lambda

    :param l: float => mean of distribution (lambda)
    :param k_vals: np.ndarray => integers to use

    :return: np.ndarray for cdf
    """
    cdf = np.zeros((len(k_vals)))
    for i in range(len(k_vals)):
        subset = np.arange(int(k_vals[i]))
        cdf[i] = np.sum((l ** subset) / scm.factorial(subset))
    cdf *= np.exp(-l)
    return cdf


def plot_cdf(data, filename, write, show):
    """
    Poisson distribution is defined as the number of events that occur in some
    time period.

    Using a specified length time bin we can define our distribution based on
    how many events occur in each bin.

    https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

    In essence, the closer our empirical distribution is to a poisson
    distribution, the closer to zero this statistic should be.

    Therefore, ideally we minimize this statistic.

    :param data: dict => 1d array of random variable values
    :param filename: str => filename to use for saving
    :param write: bool => Whether or not to save the plot
    :param show: bool => whether or not to show the plot

    :return: None
    """
    for k, v in sorted(data.items(), key=lambda tup: tup[0]):
        print('{}\t{}'.format(k, v))

    min_s = 0  # least amount of solitons to pass per slot
    max_s = 15  # most amount of solitons to pass
    k_vals = np.arange(min_s, max_s, 0.01)  # cause I like smooth lines

    plt.figure(figsize=(10, 10))
    i = 0
    k_data = np.zeros((len(data), 2))
    for pos, rv_data in sorted([(k, np.array(v)) for k, v in data.items()],
                               key=lambda tup:tup[0]):
        ecdf                    = get_empirical_cdf(rv_data, k_vals)
        l                       = np.mean(rv_data)
        cdf                     = get_poisson_cdf(l, k_vals)
        kolmogorov_smirnov_stat = (np.abs(ecdf - cdf).max())
        # plt.figure()
        # plt.plot(k_vals, ecdf, label='Emperical CDF')
        # plt.plot(k_vals, cdf, label='Poisson CDF')
        # plt.legend(loc=0)
        # plt.title(r'$\lambda={:0.03f}$, $ks={:0.03f}$'.format(l, kolmogorov_smirnov_stat))
        # plt.show()
        k_data[i]               = [pos, kolmogorov_smirnov_stat]
        i += 1

    plt.scatter(k_data[:, 0], k_data[:, 1])

    # Lines of best fit
    x = np.linspace(k_data[:, 0].min(), k_data[:, 0].max(), 1000)
    def r2_score(yhat, y):
        ss_res = 0
        for y0, y1 in zip(yhat, y):
            ss_res += (y0 - y1)**2
        ss_tot = 0
        ymean = y.mean()
        for y0 in y:
            ss_tot += (y0 - ymean)**2
        return (1 - (ss_res / ss_tot))

    ## Degree 1
    params = np.polyfit(k_data[:, 0], k_data[:, 1], 1)
    degree1 = lambda x: params[0] * x + params[1]
    plt.plot(x, degree1(x),
            label=r'Degree 1, $r^2={:0.03f}$'.format(r2_score(degree1(k_data[:, 0]), k_data[:, 1])))

    ## Degree 2
    params = np.polyfit(k_data[:, 0], k_data[:, 1], 2)
    degree2 = lambda x: params[0] * x**2 + params[1] * x + params[2]
    plt.plot(x, degree2(x),
            label=r'Degree 2, $r^2={:0.03f}$'.format(r2_score(degree2(k_data[:, 0]), k_data[:, 1])))

    ## Degree 1/2
    params = np.polyfit(k_data[:, 0], k_data[:, 1]**2, 1)
    degree1_2 = lambda x: (params[0] * x + params[1])**(1/2)
    plt.plot(x, degree1_2(x),
            label=r'Degree $1/2$, $r^2={:0.03f}$'.format(r2_score(degree1_2(k_data[:, 0]), k_data[:, 1])))

    # things
    plt.legend(loc=0)  # loc=0 is best
    plt.xlabel('Position in Conduit')
    plt.ylabel('Kolmogorov-Smirnov Statistic')

    if write:
        plt.savefig('./output/{}_cdf.png'.format(filename.split('.')[0]))

    if show:
        plt.show()


def poissonness(data: Dict[int, List[int]]) -> None:
    sorted_data = sorted(data.items(), key=lambda tup: tup[0])
    for k, v in sorted_data:
        print('{}\t{}'.format(k, v))

    errors = np.zeros(len(sorted_data))
    for i, (index, observations) in enumerate(sorted_data):
        freqs = util.get_frequencies(observations)
        ht_data = util.mass_hoaglin_tukey(freqs, observations)

        _, residuals, *_ = np.polyfit(np.arange(len(ht_data)), ht_data, 1, full=True)

        errors[i] = residuals[0]

    plt.figure()
    plt.scatter(np.arange(len(errors)), errors)
    plt.show()

    return


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='File to analyze')
    parser.add_argument('-d', '--directory', type=str, default=None,
                        help='Directory to analyze')
    parser.add_argument('-w', '--write', action='store_true', default=False,
                        help=('Save Figures'))
    parser.add_argument('-s', '--show', action='store_true', default=False,
                        help=('Show Figures'))
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help=('Plot the data'))
    args = parser.parse_args()
    if (not args.file and not args.directory) or (args.file and args.directory):
        print('File OR directory must be supplied. Not both. Not Neither.')
        sys.exit(0)
    if args.directory and not os.path.exists(args.directory):
        print('Must supply valid directory')
        sys.exit(0)
    if args.file and not os.path.exists(args.file):
        print('Must supply valid file')
        sys.exit(0)
    return args


if __name__ == '__main__':
    sys.exit(main())
