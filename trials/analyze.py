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


def main():
    args = get_args()

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

    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # first lets get all the simulationids
    c.execute('SELECT wnum FROM simulations ORDER BY wnum ASC')
    workers = [tup[0] for tup in c.fetchall()]

    wsims = []
    for w in workers:
        c.execute('SELECT id FROM simulations WHERE wnum=?', (w,))
        sims = [tup[0] for tup in c.fetchall()]

        simdata = []
        for sim in sims:
            c.execute('SELECT t, p FROM poissonness WHERE simulationid=?', (sim,))
            simdata.append(np.array(c.fetchall()))

        simdata.sort(key=lambda x: len(x))

        wsims.append(simdata)

    plt.figure(figsize=(8, 8))

    for pair in wsims:
        short = True
        for data in pair:
            plt.plot(data[:, 0], data[:, 1], ('r-' if short else 'b-'))
            short ^= True

    plt.savefig('out.png')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', metavar='D', type=str, nargs=1,
                        help='Directory to analyze')
    args = parser.parse_args()
    args.directory = args.directory[0]
    return args



if __name__ == '__main__':
    sys.exit(main())
