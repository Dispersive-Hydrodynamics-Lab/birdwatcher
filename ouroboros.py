#!/usr/bin/env python3

# TODO: Check if running every 30 or so minutes and if nothing has happened email me

import sys, os
import random
import argparse
import time
# import threading
# import queue
import multiprocessing as mp
from multiprocessing import Queue
import logging
import requests
import sqlite3 as sq
from tqdm import tqdm
from functools import reduce

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import numpy as np

import statsmodels as st
from statsmodels.distributions.empirical_distribution import ECDF

import scipy
import scipy.io as sc_io
import scipy.interpolate as sc_in
import scipy.signal as sc_sg
import scipy.stats as sc_st

import matlab
import matlab.engine

from typing import Optional, Dict, List, Any


GOAL = 0
DEATH = 'POISON PILL'
DIRSTR = ('conduit_eqtn_wnum_{}_tmax_{}_zmax_{}'
          '_Nz_{}_order_4_init_condns_soligas_'
          'iter_{}_bndry_condns_periodic')


def main():
    args = get_args()

    if args.testing:
        return

    logging.getLogger('ouroboros')
    logging.basicConfig(filename='ouroboros.log',
                        level=logging.INFO,
                        format='%(asctime)s \t %(message)s')
    logging.info('~~~STARTING~~~')
    logging.info(str(args))

    # parse_queue    = queue.Queue()
    # write_queue    = queue.Queue()
    # MATLAB_queue   = queue.Queue()
    # deletion_queue = queue.Queue()
    # done_queue     = queue.Queue()
    parse_queue    = Queue()
    write_queue    = Queue()
    MATLAB_queue   = Queue()
    deletion_queue = Queue()
    done_queue     = Queue()

    # don't ask why this is here
    sim = RunSimulation(MATLAB_queue, args.num_simulations)

    print('(1/5) Starting Filesystem Observer')
    handler = SimulationHandler(parse_queue)
    observer = Observer()
    observer.schedule(handler, args.directory, recursive=True)
    observer.start()

    print('(2/5) Starting Database')
    db = Database(args.database, write_queue, MATLAB_queue, sim, done_queue)
    db.start()

    print('(3/5) Starting Workers')
    workers = []
    for i in range(args.num_workers):
        parser = ParseFile(parse_queue, write_queue, deletion_queue, i)
        parser.start()
        workers.append(parser)

    print('(4/5) Starting Simulations')
    sim.start()

    print('(5/5) Starting Deleter')
    deleter = Deleter(deletion_queue)
    deleter.start()

    # Start the initial sims
    for i in range(args.num_simulations):
        initial_profile_L1 = ParseFile.generate_initial_profile(domain_size=500, number_of_solitons=16, wnum=i)
        initial_profile_L2 = ParseFile.generate_initial_profile(domain_size=1000, number_of_solitons=32, wnum=i)
        MATLAB_queue.put(('run{}'.format(i),
                         (initial_profile_L1, initial_profile_L2)))

    try:
        print('Sleeping')
        while True:
            time.sleep(1)
            done = done_queue.get()
            if done == DEATH:
                print('finished')
                break
    except KeyboardInterrupt:
        logging.info('~~~CLOSING~~~')

    print('(1/5) Closing Simulations')
    MATLAB_queue.put(DEATH)
    sim.join()
    print('(2/5) Closing Workers')
    for i in range(args.num_workers):
        parse_queue.put(DEATH)
    for worker in workers:
        worker.join()
    print('(3/5) Closing Database')
    write_queue.put(DEATH)
    db.join()
    print('(4/5) Closing Observer')
    observer.stop()
    observer.join()
    print('(5/5) Closing Deleter')
    deletion_queue.put(DEATH)
    deleter.join()


class SimulationHandler(FileSystemEventHandler):
    def __init__(self, q, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files_to_parse = q

    def on_created(self, event):
        if ((event.src_path[-4:] == '.mat') and
                ('parameters' not in event.src_path) and
                ('/00000.mat' not in event.src_path)):
            time.sleep(0.1)  # errors when reading file before finished
            logging.info('HANDLER\t{} created. Adding to Queue'.format(event.src_path))
            self.files_to_parse.put(event.src_path)


# class Deleter(threading.Thread):
class Deleter(mp.Process):
    def __init__(self, q, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_queue = q
        self.files_for_deletion = {}

    def run(self):
        while True:
            message = self.input_queue.get()
            if message == DEATH:
                break

            sim_name = os.path.basename(os.path.dirname(message))
            if sim_name not in list(self.files_for_deletion.keys()):
                self.files_for_deletion[sim_name] = [os.path.basename(message)]
            else:
                self.files_for_deletion[sim_name].append(os.path.basename(message))
                for file in sorted(self.files_for_deletion[sim_name][:-1], key=lambda s: int(s.split('.')[0])):
                    os.remove(os.path.join(os.path.dirname(message), file))
                self.files_for_deletion[sim_name] = [self.files_for_deletion[sim_name][-1]]


# class ParseFile(threading.Thread):
class ParseFile(mp.Process):
    def __init__(self, q0, q1, q2, i, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files_to_parse = q0
        self.write_queue    = q1
        self.deletion_queue = q2
        self.worker_num     = i

    def log(self, message):
        logging.info('PARSER{}\t{}'.format(self.worker_num, message))

    def run(self, *args, **kwargs):
        while True:
            next_file = self.files_to_parse.get()
            self.log('Received {}'.format(next_file))
            if next_file == DEATH:
                self.log('Sent `DEATH`, dying gracefully.')
                return
            self.parse_file(next_file)
            self.deletion_queue.put(next_file)

    def parse_file(self, filename):
        # First let's load the parameters if they haven't loaded already
        file_directory = os.path.dirname(filename)
        parameters = sc_io.loadmat(os.path.join(file_directory, 'parameters.mat'))
        # Now load in the file that was loaded
        datafile = sc_io.loadmat(filename)
        data = {
            'zmax': parameters['zmax'][0][0],
            't': parameters['t'][0],
            'data': datafile['A'].T[0],
            'domain': np.arange(0,
                                parameters['zmax'][0][0],
                                parameters['dz'][0][0]),
            'tnow': datafile['tnow'][0][0],
            'wnum': parameters['worknum'][0][0]
        }
        self.write_queue.put(('init', (file_directory, data['zmax'], data['t'], data['wnum'])))
        peaks = self.get_peaks(data['data'], data['domain'])
        self.write_queue.put(('peaks', (peaks, data['tnow'], file_directory)))
        poissonness, ecdf = self.get_poissonness(peaks)
        self.write_queue.put(('poissonness', (poissonness, data['tnow'], file_directory, ecdf)))

    def get_poissonness(self, peaks):
        """
        Need to look at gaps
        """
        gaps = peaks[1:, 0] - peaks[:-1, 0]  # this is our RV
        λ = gaps.mean()  # estimator of true mean
        prob_vals = np.linspace(1e-5, 1 - 1e-5, 30)   # huehue shitty number, adjusted for no infs
        # This gets empirical quantiles from dataset
        gap_quantiles = sc_st.mstats.mquantiles(gaps, prob=prob_vals)
        # Now let's compare vs. theoretical quantiles
        # https://en.wikipedia.org/wiki/Exponential_distribution#Quantiles
        theoretical_quantiles = -np.log(1 - prob_vals) / λ
        poissonness = np.sum((gap_quantiles - theoretical_quantiles)**2)  # RSS
        return poissonness, ECDF(gaps)

    def get_peaks(self, data, domain):
        thresh = 1.1  # TODO: Make this adjustable - issue#2
        peakrange = np.arange(1, 100)
        rawpeaks = np.array(sc_sg.find_peaks_cwt(data, peakrange))
        rawpeaks = rawpeaks[rawpeaks > rawpeaks.min()]  # Remove leftmost peak (false positive at input)
        threshpeaks = rawpeaks[data[rawpeaks] > thresh]
        return np.array([domain[threshpeaks], data[threshpeaks]]).T

    @staticmethod
    def generate_initial_profile_by_cdf(domain_size: int, cdf, iter_num: int, tmax: int, wnum: int):
        H = 0.01  # Precision of domain
        domain = np.arange(0, domain_size, H)
        soliton_positions = ParseFile.sim_exponential(cdf, domain_size)
        number_of_solitons = len(soliton_positions)
        soliton_heights = ParseFile.get_heights(number_of_solitons)
        profile = np.ones(len(domain))
        for position, height in zip(soliton_positions, soliton_heights):
            soliton = ParseFile.get_soliton(domain, height, position)
            profile += soliton
        derivatives = ParseFile.get_derivative(profile, H)
        output = {
            'domain': domain,
            'area': profile,
            'area_derivative': derivatives,
            'zmax': domain_size,
            'iter': iter_num,
            'tmax': tmax,
            'wnum': wnum,
            'Nz': int(domain_size / H)
        }
        return output

    @staticmethod
    def sim_exponential(cdf, domain_size):
        domain = np.arange(0, 100, 0.01)
        cdf_vals = np.array(cdf(domain))
        cursor = 25
        spots = []
        while cursor < domain_size - 25:
            new_val = domain[cdf_vals <= random.random()][-1]
            if new_val > 10:
                cursor += new_val
                spots.append(cursor)
        return spots

    @staticmethod
    def generate_initial_profile(domain_size: Optional[int]=500,
                                 wnum: Optional[int]=1,
                                 number_of_solitons: Optional[int]=16,
                                 iter_num: Optional[int]=1,
                                 tmax: Optional[int]=600) -> Dict[str, np.ndarray]:
        H = 0.01  # Precision of domain
        domain = np.arange(0, domain_size, H)
        soliton_heights = ParseFile.get_heights(number_of_solitons)
        soliton_positions = np.linspace(25, domain_size - 25, number_of_solitons)
        profile = np.ones(len(domain))
        for position, height in zip(soliton_positions, soliton_heights):
            soliton = ParseFile.get_soliton(domain, height, position)
            profile += soliton
        derivatives = ParseFile.get_derivative(profile, H)
        output = {
            'domain': domain,
            'area': profile,
            'area_derivative': derivatives,
            'zmax': domain_size,
            'iter': iter_num,
            'tmax': tmax,
            'wnum': wnum,
            'Nz': int(domain_size / H)
        }
        return output

    @staticmethod
    def get_heights(count):
        pre_comp = np.arange(2.0, 7.5, 0.5)
        heights = np.random.choice(np.arange(1, 5.1, 0.5),
                                   size=count,
                                   replace=True)
        return heights

    @staticmethod
    def get_soliton(domain: np.ndarray, amplitude: int, position: int) -> np.ndarray:
        soliton = sc_io.loadmat('./solitons/conduit{:g}.mat'.format(amplitude))
        interpolation = sc_in.InterpolatedUnivariateSpline(soliton['z'], soliton['phi'])
        if np.abs(interpolation(domain - position) > 20).any():
            return get_soliton(domain, amplitude + 1, position)
        return interpolation(domain - position)

    @staticmethod
    def get_derivative(f: np.ndarray, H: float) -> np.ndarray:
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


# class Database(threading.Thread):
class Database(mp.Process):
    def __init__(self, filename, q0, q1, sim, done, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename     = filename
        self.write_queue  = q0
        self.MATLAB_queue = q1
        self.done_queue   = done
        self.sim          = sim
        self._init_db()

    def log(self, message):
        logging.info('DATABASE\t{}'.format(message))

    def _init_db(self):
        conn = sq.connect(self.filename)
        c = conn.cursor()
        c.execute(('CREATE TABLE IF NOT EXISTS simulations('
                        'id INTEGER PRIMARY KEY,'
                        'wnum INTEGER,'
                        'directory TEXT)'))
        c.execute(('CREATE TABLE IF NOT EXISTS parameters('
                        'id INTEGER PRIMARY KEY,'
                        'simulationid INT,'
                        'zmax REAL,'
                        'FOREIGN KEY(simulationid) REFERENCES simulations(id))'))
        c.execute(('CREATE TABLE IF NOT EXISTS tvals('
                        'id INTEGER PRIMARY KEY,'
                        'simulationid INT,'
                        'tval REAL,'
                        'FOREIGN KEY(simulationid) REFERENCES simulations(id))'))
        c.execute(('CREATE TABLE IF NOT EXISTS peaks('
                        'id INTEGER PRIMARY KEY,'
                        'simulationid INT,'
                        'x REAL,'
                        'y REAL,'
                        't REAL,'
                        'FOREIGN KEY(simulationid) REFERENCES simulations(id))'))
        c.execute(('CREATE TABLE IF NOT EXISTS poissonness('
                        'id INTEGER PRIMARY KEY,'
                        'simulationid INT,'
                        't REAL,'
                        'p REAL,'
                        'FOREIGN KEY(simulationid) REFERENCES simulations(id))'))
        conn.commit()
        conn.close()
        self.log('Tables Created')

    def run(self, *args, **kwargs):
        conn = sq.connect(self.filename)
        while True:
            # message should always be a tuple
            message = self.write_queue.get()
            if message == DEATH:
                self.log('Dying happily')
                conn.close()
                break

            if message[0] == 'init':
                c = conn.cursor()
                directory, zmax, t, wnum = message[1]
                directory = os.path.basename(directory)
                c.execute('SELECT COUNT(*) from simulations WHERE directory=?', (directory,))
                results = c.fetchall()
                if results[0][0] == 0:
                    c.execute('INSERT INTO simulations(wnum, directory) VALUES(?,?)', (int(wnum), directory))
                    c.execute('SELECT id FROM simulations WHERE directory=?', (directory,))
                    sim_id = c.fetchall()[0][0]
                    c.execute('INSERT INTO parameters(simulationid, zmax) VALUES(?,?)',
                              (sim_id, float(zmax)))
                    c.executemany('INSERT INTO tvals(simulationid, tval) VALUES(?,?)',
                                  [(sim_id, _) for _ in t])
                conn.commit()
                c.close()

            if message[0] == 'peaks':
                c = conn.cursor()
                peaks, tnow, directory = message[1]
                directory = os.path.basename(directory)
                c.execute('SELECT id FROM simulations WHERE directory=?', (directory,))
                sim_id = c.fetchall()[0][0]
                c.executemany('INSERT INTO peaks(simulationid, x, y, t) VALUES (?, ?, ?, ?)',
                              [(sim_id, row[0], row[1], float(tnow)) for row in peaks])
                conn.commit()
                c.close()

            if message[0] == 'poissonness':
                # first add new
                c = conn.cursor()
                poissonness, tnow, directory, ecdf = message[1]
                directory = os.path.basename(directory)
                c.execute('SELECT id FROM simulations WHERE directory=?', (directory,))
                sim_id = c.fetchall()[0][0]
                c.execute('INSERT INTO poissonness(simulationid, t, p) VALUES(?,?,?)',
                          (sim_id, float(tnow), poissonness))
                conn.commit()
                # Now see how we are next to goal
                c.execute(
                    'SELECT MAX(t), simulationid '
                    'FROM poissonness '
                    'GROUP BY simulationid '
                    'ORDER BY simulationid DESC '
                    'LIMIT (SELECT COUNT(DISTINCT wnum) FROM simulations) * 2'
                )
                if reduce(lambda acc, n: acc and (n > GOAL), [_[0] for _ in c.fetchall()], True):
                    self.done_queue.put(DEATH)
                    c.close()
                    conn.close()
                    break
                # Now check to see if we're past threshold (arbitrary 10%)
                # First get the min of the two maxtimes and their corresponding p vals
                bothdirs = self.sim.get_both_dirnames(os.path.basename(directory))
                if bothdirs is not None:
                    dirs = {d: 0 for d in bothdirs}
                    abort_flag = False
                    for d in dirs:
                        c.execute('SELECT id FROM simulations WHERE directory=?', (d,))
                        result = c.fetchall()
                        if len(result) == 0:
                            abort_flag = True
                            break
                        else:
                            sim_id = result[0][0]
                        c.execute('SELECT MAX(t), p FROM poissonness WHERE simulationid=?', (sim_id,))
                        result = c.fetchall()
                        if len(result) != 0:
                            dirs[d] = result[0]
                    for d, (t, p) in dirs.items():
                        if t is None or p is None:
                            abort_flag = True
                    if not abort_flag:
                        mintime = min(t for _, (t, _) in dirs.items())
                        for d, (t, p) in dirs.items():
                            c.execute('SELECT id FROM simulations WHERE directory=?', (d,))
                            sim_id = c.fetchall()[0][0]
                            if t != mintime:
                                c.execute('SELECT t, p FROM poissonness WHERE simulationid=? AND t=?',
                                          (sim_id, mintime))
                                dirs[d] = c.fetchall()[0]
                        # Now we can compare
                        pvals = [p for _, (_, p) in dirs.items()]
                        if np.abs((pvals[0] / pvals[1]) - 1) > 0.2:
                            # Stop
                            self.MATLAB_queue.put(('KILL BY DIR', list(dirs.keys())[0]))
                            # and restart
                            info = []
                            for d in dirs:
                                c.execute(('SELECT simulations.id, parameters.zmax, simulations.wnum '
                                               'FROM simulations '
                                               'INNER JOIN parameters '
                                               'ON simulations.id = parameters.simulationid '
                                               'WHERE simulations.directory=?'), (d,))
                                info.append(c.fetchall()[0])
                            old_2l_zmax = max(info[0][1], info[1][1])
                            old_run_num = max(info[0][0], info[1][0])
                            initial_profile_L1 = ParseFile.generate_initial_profile_by_cdf(old_2l_zmax,
                                                                                           ecdf,
                                                                                           old_run_num + 1,
                                                                                           info[0][1]**2,
                                                                                           info[0][2])
                            initial_profile_L2 = ParseFile.generate_initial_profile_by_cdf(old_2l_zmax * 2,
                                                                                           ecdf,
                                                                                           old_run_num + 2,
                                                                                           info[0][1]**2,
                                                                                           info[0][2])
                            print('~~~~~~~~~~~~~~~~~~~~~~~ROTATING~~~~~~~~~~~~~~~~~~~~~~~')
                            self.MATLAB_queue.put(('runN', (initial_profile_L1, initial_profile_L2)))
                c.close()



# class RunSimulation(threading.Thread):
class RunSimulation(mp.Process):
    def __init__(self, input_queue, num_simulations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_queue = input_queue
        self.num_simulations = num_simulations
        self.jobs = {}

    def get_both_dirnames(self, dir1):
        dirs = None
        for runname, jobs in self.jobs.items():
            if dir1 in list(jobs.keys()):
                dirs = list(jobs.keys())
                break
        return dirs

    def run(self):
        while True:
            message = self.command_queue.get()
            if message == DEATH:
                for name, jobs in self.jobs.items():
                    for jobname, (job, q) in jobs.items():
                        q.put(DEATH)
                        job.join()
                break
            if message[0] == 'KILL':
                run_name = message[1]
                for jobname, (job, q) in self.jobs[run_name].items():
                    q.put(DEATH)
            elif message[0] == 'KILL BY DIR':
                dirname = message[1]
                for runname, jobs in self.jobs.items():
                    if dirname in list(jobs.keys()):
                        run_name = runname
                        break
                for jobname, (job, q) in self.jobs[run_name].items():
                    q.put(DEATH)
            else:
                name = message[0]
                self.jobs[name] = {}
                for profile in message[1]:
                    # message_queue = queue.Queue()
                    message_queue = Queue()
                    j = MATLABScript(message_queue,
                                     (float(profile['iter']),
                                      float(profile['wnum']),
                                      float(profile['zmax']),
                                      float(profile['tmax']),
                                      matlab.double(profile['domain'].tolist()),
                                      matlab.double(profile['area'].tolist())))
                    j.start()
                    jobname = DIRSTR.format(
                            profile['wnum'],
                            profile['tmax'],
                            profile['zmax'],
                            profile['Nz'],
                            profile['iter'])
                    self.jobs[name][jobname] = (j, message_queue)


# class MATLABScript(threading.Thread):
class MATLABScript(mp.Process):
    def __init__(self, message_queue, fargs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_queue = message_queue
        self.args = fargs

    def run(self):
        eng = matlab.engine.start_matlab()
        job = eng.driver_conduit_solver_will(*self.args,
                                             nargout=0,
                                             async=True)
        while True:
            message = self.message_queue.get()
            if message == DEATH:
                job.cancel()
                eng.quit()
                break


def get_args():
    global GOAL
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', metavar='d', type=str, nargs='*',
                        default='.',
                        help='Directory to watch for simulation data (recursive)')
    parser.add_argument('-n', '--num-workers', type=int, default=8,
                        help='Number of parse workers to use.')
    parser.add_argument('-s', '--num-simulations', type=int, default=2,
                        help='How many MATLAB sims to run')
    parser.add_argument('-db', '--database', type=str, default='ouroboros.db',
                        help='Database file to use.')
    parser.add_argument('-t', '--testing', action='store_true', default=False,
                        help='Run in test mode (i.e. immediately exit)')
    parser.add_argument('-g', '--goal', type=int, default=100,
                        help='Goal time to reach. When ANY sim hits this the entire thing stops.')
    args = parser.parse_args()
    if isinstance(args.directory, list):
        args.directory = args.directory[0]
    GOAL = args.goal
    return args


if __name__ == '__main__':
    sys.exit(main())
