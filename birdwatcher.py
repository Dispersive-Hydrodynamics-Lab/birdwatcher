#!/usr/bin/env python3
"""
Checks if distribution is Poisson

There are a couple of major points with this

1. Identifying the "peaks" of the solitons
2. Determining the best time bin size (this is gonna be trial/error)
3. Using the Kolmogorov-Smirnov Statistic to determine how similar the data is
    to a Poisson Distribution. (DONE)
"""

import sys
import os
import argparse
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io as sco
import scipy.misc as scm
import scipy.ndimage.filters as scfilt
import scipy.signal as scsig
from tqdm import tqdm


def main():
    """ Handle program execution """
    args = get_args()
    for filename in args.files:
        print('Processing {}\n'.format(filename) + ('=' * (len(filename) + 11)))
        rawdata  = load_data(filename)
        data     = rawdata['right_edges'] - rawdata['left_edges']
        savename = os.path.basename(filename).split('.')[0]
        phase_portrait(data, args.write, args.show, savename)
        peaks, solitons, soliton_hash = plot_solitons(data, savename,
                                                      show=args.show,
                                                      write=args.write)
        rv = solitons_to_rv(solitons, soliton_hash, len(data[0]), len(data))
        plot_cdf(rv, savename, args.write, args.show)


def solitons_to_rv(solitons, soliton_hash, maxtime, maxpos, binsize=20):
    """
    Converts the solitons that have been found to a random variable.

    :param solitons: dictionary => soliton name to array of times and positions
    :param soliton_hash: dictionary => soliton name to dictionary of times to positions
    :param maxtime: int => the max "time" (width of phase portrait)
    :param maxpos: int => the max "position" (height of phase portrait)
    :param binsize: (int) => how many seconds to elapse before recording value

    :returns: np.ndarray => 1d array of values
    """
    position = 0.25 * maxpos
    crossed = 0
    rv = []
    for i in range(0, maxtime, binsize):
        count = 0
        for soliton, indices in soliton_hash.items():
            try:
                cpos = indices[i][0]
            except KeyError:
                cpos = maxpos
            if cpos <= position:
                count += 1
        rv.append(count - crossed)
        crossed += count
    return np.array(rv)


def phase_portrait(data, write, show, filename):
    """
    Plots phase portrait for current dataset

    :param data: np.ndarray => 2d array of values
    :param write: bool => Whether or not to save the plot
    :param show: bool => whether or not to show the plot
    :param filename: str => the savename of the file

    :return: None
    """
    print('Plotting Phase Portrait')
    if write:
        plt.imsave('./output/{}_phaseportrait.png'.format(filename),
                   scm.imresize(data, (1000, 1000)))

    if show:
        plt.figure()
        plt.imshow(scm.imresize(data, (1000, 1000)))
        plt.show()


def plot_solitons(data, filename, show=False, write=False):
    """
    Plot (and animate) the solitons.

    :param data: np.ndarray => 2d array of values
    :param filename: str => the savename of the file
    :param show: bool => whether or not to show the plot
    :param write: bool => Whether or not to save the plot

    :return peaks: list => contains a 2d np.ndarray for every frame
    :return solitons: dictionary => soliton name to array of times and positions
    :return soliton_hash: dictionary => soliton name to dictionary of times to positions
    """
    # Initialize datapoints
    domain = np.arange(len(data))
    cdata = data[:, 0]

    # Initialize smoothed data
    print('Smoothing Data')
    sdata = scfilt.gaussian_filter1d(data, 50, axis=0)
    scdata = sdata[:, 0]

    # Establish figure
    print('Initializing Figure')
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # Plot Raw Data
    line, = ax.plot(domain, scdata, 'b-')
    # Plot Smoothed Data
    line1, = ax.plot(domain, cdata, 'b-', alpha=0.5)

    vert_domain = np.arange(0, 250)
    ax.plot(len(data) * 0.25 * np.ones(len(vert_domain)), vert_domain, 'r-')

    # Set figure axes limits
    ax.set_ylim(0, 250)
    ax.set_xlim(0, domain.max())

    # Grab and plot peaks
    peaks = get_peaks(domain, sdata)

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

    def animate(i):
        """ Animation function for plot """
        cdata = data[:, i]   # New datapoints
        scdata = sdata[:, i] # New Smoothed datapoints
        line.set_ydata(scdata)
        line1.set_ydata(cdata)
        points.set_data(peaks[i][:, 0], peaks[i][:, 1])

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
                                  interval=180)

    print('Animating')
    if write:
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Will Farmer'), bitrate=1800)
        anim_name = './output/{}_solitons.mp4'.format(filename)
        ani.save(anim_name, writer=writer)
        os.system('ffmpeg -i "{}" "{}"'.format(anim_name,
                                               '.'.join(anim_name.split('.')[:-1]) + '.gif'))
        os.system('rm {}'.format(anim_name))
    if show:
        plt.show()

    return peaks, solitons, soliton_hash


def peaks_func(d, domain):
    """
    Determine peaks for frame

    :param d: np.ndarray => 1d array of y values
    :param domain: np.ndarray => 1d array of x values

    :return: np.ndarray => array of XY coordinates of the positions
    """
    thresh = 120  # TODO: Make this adjustable
    peakrange = np.arange(1, 100)
    rawpeaks = np.array(scsig.find_peaks_cwt(d, peakrange))
    rawpeaks = rawpeaks[rawpeaks < rawpeaks.max()]  # Remove rightmost peak (false positive)
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
        futures = {executor.submit(peaks_func, sdata[:, i], domain):i for i in range(len(sdata[0]))}
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

        for csoli in cstate:                      # For each point in the current state
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
                soliton_hash[nkey] = {i:nsoli}
    return solitons, soliton_hash


def load_data(filename):
    """
    Loads .mat datafile.

    TODO: Error Handling

    :param filename: str => file to load

    :return: dictionary of np.ndarrays
    """
    return sco.loadmat(filename)


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
        cdf[i] = np.sum((l**subset) / scm.factorial(subset))
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

    :param data: np.ndarray => 1d array of random variable values
    :param filename: str => filename to use for saving
    :param write: bool => Whether or not to save the plot
    :param show: bool => whether or not to show the plot

    :return: None
    """
    min_s = 0  # least amount of solitons to pass per slot
    max_s = 10 # most amount of solitons to pass
    k_vals = np.arange(min_s, max_s, 0.01)   # cause i like smooth lines

    print('Finding Empirical CDF')
    ecdf = get_empirical_cdf(data, k_vals)

    l = np.mean(data)

    print('Finding Poisson CDF')
    cdf = get_poisson_cdf(l, k_vals)

    kolmogorov_smirnov_stat = (np.abs(ecdf - cdf).max())

    print('Saving CDF Analysis')
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(k_vals, ecdf, 'r-', label='Empirical CDF')
    ax.plot(k_vals, cdf, 'b-', label='Poisson CDF')
    ax.fill_between(k_vals, ecdf, cdf, alpha=0.2)
    ax.legend()
    ax.set_title(r'Kolmogorov-Smirnov Statistic: ${0:.5f}$'.format(kolmogorov_smirnov_stat))

    if write:
        plt.savefig('./output/{}_cdf.png'.format(filename.split('.')[0]))

    if show:
        plt.show()


def get_args():
    """
    Get arguments

    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='F', type=str, nargs='*',
                        help=('File(s) for processing.'))
    parser.add_argument('-g', '--generate', action='store_true', default=False,
                        help=('Generate a .mat file with data'))
    parser.add_argument('-w', '--write', action='store_true', default=False,
                        help=('Save Figures'))
    parser.add_argument('-s', '--show', action='store_true', default=False,
                        help=('Show Figures'))
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help=('Plot the data'))
    args = parser.parse_args()
    try:
        assert len(args.files) != 0
    except AssertionError:
        print('Missing Files')
    return args


if __name__ == '__main__':
    sys.exit(main())
