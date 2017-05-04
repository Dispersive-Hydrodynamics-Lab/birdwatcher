#!/usr/bin/env python3.6

"""
    Make the oscillating-type plot for all of the runs
    See if there is a consistent 'period' of error oscillation for a domain with fixed zmax
    Fix your code so it'll go to the next level :)
    Make histograms or some other stats-graph of the soliton amplitudes and centers--see if the ampl. dist. is conserved and if the centers really start looking Poisson.

Thanks!
Michelle
"""

import sys
import os
import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import re


def main():
    args = get_args()

    # find db in directory provided
    db_file = None
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file == 'ouroboros.db':
                db_file = os.path.join(root, file)
                break
        if db_file is not None:
            break
    else:
        raise FileNotFoundError('No database file')

    # connect to db
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    ####################################################################################################################
    # POISSONNESS OVER TIME
    ####################################################################################################################
    # let's first get all the distinct worker nums
    c.execute(
        """
        SELECT
            A.simulationid, B.wnum, B.directory, A.t, A.p
        FROM poissonness AS A
        INNER JOIN simulations AS B ON A.simulationid = B.id
        ORDER BY A.simulationid, B.wnum ASC;
        """
    )
    data = c.fetchall()

    # convert to dataframe
    data = pd.DataFrame(data, columns=['id', 'worker', 'name', 'time', 'poisval'])

    # get plotcount
    plotcount = len(data['worker'].unique())

    # pull out the useful part of the name
    data['name'] = data['name'].apply(lambda x: re.findall('^.*(zmax_[0-9]+).*$', x)[0])

    groups = data.\
        groupby(['worker', 'id'])

    # plot each group on plot corresponding to wnum & simid
    # MAY HAVE TO TWEAK FIGSIZE EACH TIME
    fig, axarr = plt.subplots(plotcount, 1, figsize=(8, 16))
    for key, group in groups:
        axarr[key[0]].plot(group['time'], group['poisval'], label=group['name'].iloc[0])
        axarr[key[0]].set_xlabel('Timestep of Simulation')
        axarr[key[0]].set_ylabel('Error')
        axarr[key[0]].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('poissonness.png')

    ####################################################################################################################
    # AMPLITUDES OVER TIME
    ####################################################################################################################
    c.execute("""
        SELECT
            A.simulationid, B.wnum, B.directory, A.t, A.x, A.y
        FROM peaks AS A
        INNER JOIN simulations AS B
        ON A.simulationid = B.id
        ORDER BY A.simulationid, B.wnum ASC;
    """)
    data = c.fetchall()

    data = pd.DataFrame(data, columns=['id', 'worker', 'name', 't', 'x', 'y'])

    # pull out the useful part of the name
    data['name'] = data['name'].apply(lambda x: re.findall('^.*(zmax_[0-9]+).*$', x)[0])

    groups = data.groupby(['worker', 'id'])

    fig, axarr = plt.subplots(plotcount * 2, 1, figsize=(8, 32))
    hist_bins = np.linspace(1, 7, 10)
    flag = False
    for key, group in groups:
        segments = group.groupby(['t'])
        array = np.zeros((len(hist_bins) - 1, len(segments)))
        for i, (tval, segment) in enumerate(segments):
            hist_vals = np.histogram(segment.y.values, bins=hist_bins)[0]
            array[:, i] = hist_vals

        index = 2 * key[0]
        if flag:
            index = 2 * key[0] + 1
            flag = False
        else:
            flag = True
        axarr[index].imshow(array, interpolation='nearest')

        axarr[index].axis('off')

    plt.tight_layout()
    plt.savefig('center_histograms.png')



    ####################
    c.execute("""
        SELECT A.simulationid, A.t, A.p, B.p, ABS(A.p - B.p) / MIN(A.p, B.p)
        FROM (SELECT * FROM poissonness WHERE simulationid=1) AS A
        INNER JOIN (SELECT * FROM poissonness WHERE simulationid=3) AS B
        ON A.t = B.t
    """)
    data = pd.DataFrame(c.fetchall(), columns=['id', 't', 'Ap', 'Bp', 'error'])

    data.plot(x='t', y='error')
    plt.show()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', metavar='D', type=str, nargs=1,
                        help='Directory to analyze')
    args = parser.parse_args()
    args.directory = args.directory[0]
    return args


if __name__ == '__main__':
    sys.exit(main())
