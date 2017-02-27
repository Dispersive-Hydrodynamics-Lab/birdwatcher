#!/usr/bin/env python3

# TODO: Check if running every 30 or so minutes and if nothing has happened email me

import sys, os
import argparse
import time
import threading
import queue
import logging
import requests
import sqlite3 as sq
from tqdm import tqdm

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import numpy as np

import scipy
import scipy.io as sc_io
import scipy.interpolate as sc_in

import matlab
import matlab.engine

from typing import Optional, Dict, List, Any


DEATH = 'POISON PILL'
DIRSTR = ('conduit_eqtn_tmax_{}_zmax_{}'
          '_Nz_order_4_init_condns_soligas_'
          'iter_{}_bndary_condns_periodic')


def main():
    args = get_args()

    logging.getLogger('ouroboros')
    logging.basicConfig(filename='ouroboros.log',
                        level=logging.INFO,
                        format='%(asctime)s \t %(message)s')
    logging.info('~~~STARTING~~~')
    logging.info(str(args))

    parse_queue  = queue.Queue()
    write_queue  = queue.Queue()
    db_queue     = queue.Queue()
    MATLAB_queue = queue.Queue()

    print('(1/4) Starting Filesystem Observer')
    handler = SimulationHandler(parse_queue)
    observer = Observer()
    observer.schedule(handler, args.directory, recursive=True)
    observer.start()

    print('(2/4) Starting Database')
    db = Database(args.database, write_queue, db_queue)
    db.start()

    print('(3/4) Starting Workers')
    workers = []
    for i in range(args.num_workers):
        parser = ParseFile(parse_queue, write_queue, db_queue, i)
        parser.start()
        workers.append(parser)

    print('(4/4) Starting Simulations')
    sim = RunSimulation(MATLAB_queue, args.num_simulations)
    sim.start()

    # Start the initial sims
    initial_profile_L1 = ParseFile.generate_initial_profile(domain_size=500, number_of_solitons=16)
    initial_profile_L1['iter'] = 1
    initial_profile_L1['tmax'] = 600
    initial_profile_L2 = ParseFile.generate_initial_profile(domain_size=1000, number_of_solitons=32)
    initial_profile_L2['iter'] = 1
    initial_profile_L2['tmax'] = 600
    MATLAB_queue.put(initial_profile_L1)
    MATLAB_queue.put(initial_profile_L2)

    try:
        print('Sleeping')
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info('~~~CLOSING~~~')
        print('(1/4) Closing Simulations')
        MATLAB_queue.put(DEATH)
        print('(2/4) Closing Workers')
        for i in range(args.num_workers):
            parse_queue.put(DEATH)
        print('(3/4) Closing Database')
        write_queue.put(DEATH)
        print('(4/4) Closing Observer')
        observer.stop()

    sim.join()
    for worker in workers:
        worker.join()
    db.join()
    observer.join()

    return  # let's not get willy nilly with emails yet....
    print('Sending exit email')
    response = requests.post(
        "https://api.mailgun.net/v3/sandbox25180cb882f84eaa8ba60b5dfd6572f7.mailgun.org/messages",
        auth=("api", "key-4d2898c04ec3e73b75567539a49d2074"),
        data={"from": "Mailgun Sandbox <postmaster@sandbox25180cb882f84eaa8ba60b5dfd6572f7.mailgun.org>",
              "to": "William Farmer <will@nidhogg.io>",
              "subject": "WATCHER EXIT",
              "text": "PROGRAM QUIT (HOPEFULLY THAT WAS GOOD)"})
    logging.info('SENT EMAIL WITH STATUS: {}'.format(response.status_code))


class SimulationHandler(FileSystemEventHandler):
    def __init__(self, q, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files_to_parse = q

    def on_created(self, event):
        if event.src_path[-4:] == '.mat':
            logging.info('HANDLER\t{} created. Adding to Queue'.format(event.src_path))
            self.files_to_parse.put(event.src_path)


class ParseFile(threading.Thread):
    def __init__(self, q0, q1, q2, i, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files_to_parse = q0
        self.write_queue = q1
        self.db_queue = q2
        self.worker_num = i

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

    def parse_file(self, filename):
        # First let's load the parameters if they haven't loaded already
        file_directory = os.path.dirname(filename)
        parameter_file = sc_io.loadmat(os.path.join(file_directory, 'parameters.mat'))
        # Now load in the file that was loaded
        datafile = sc_io.loadmat(filename)
        data = {
            'zmaz': parameters['zmax'],
            't': parameters['t'],
            'data': datafile['A']
        }
        return

    def get_peaks(self):
        thresh = 1.1  # TODO: Make this adjustable - issue#2
        peakrange = np.arange(1, 100)
        rawpeaks = np.array(scsig.find_peaks_cwt(d, peakrange))
        rawpeaks = rawpeaks[rawpeaks > rawpeaks.min()]  # Remove leftmost peak (false positive at input)
        threshpeaks = rawpeaks[d[rawpeaks] > thresh]
        return np.array([domain[threshpeaks], d[threshpeaks]]).T

    @staticmethod
    def generate_initial_profile(domain_size: Optional[int]=500, number_of_solitons: Optional[int]=16) -> Dict[str, np.ndarray]:
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
            'zmax': domain_size
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


class Database(threading.Thread):
    def __init__(self, filename, q0, q1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename
        self.write_queue = q0
        self.db_queue = q1
        self._init_db()

    def log(self, message):
        logging.info('DATABASE\t{}'.format(message))

    def _init_db(self):
        conn = sq.connect(self.filename)
        c = conn.cursor()
        c.execute(('CREATE TABLE IF NOT EXISTS simulations('
                        'id INT PRIMARY KEY,'
                        'name TEXT,'
                        'directory TEXT)'))
        c.execute(('CREATE TABLE IF NOT EXISTS parameters('
                        'id INT PRIMARY KEY,'
                        'simulationid INT,'
                        'zmax DOUBLE,'
                        't DOUBLE,'
                        'FOREIGN KEY(simulationid) REFERENCES simulations(id))'))
        c.execute(('CREATE TABLE IF NOT EXISTS solitons('
                        'id INT PRIMARY KEY,'
                        'simulationid INT,'
                        'parameterid INT,'
                        'x DOUBLE,'
                        'y DOUBLE,'
                        't DOUBLE,'
                        'FOREIGN KEY(simulationid) REFERENCES simulations(id),'
                        'FOREIGN KEY(parameterid) REFERENCES parameters(id))'))
        c.execute(('CREATE TABLE IF NOT EXISTS poissonness('
                        'id INT PRIMARY KEY,'
                        'simulationid INT,'
                        'p DOUBLE,'
                        'FOREIGN KEY(simulationid) REFERENCES simulations(id))'))
        conn.commit()
        conn.close()
        self.log('Tables Created')

    def run(self, *args, **kwargs):
        while True:
            # message should always be a tuple
            message = self.write_queue.get()
            if message == DEATH:
                self.log('Dying happily')
                break

            if message[0] == 'query':
                pass


class RunSimulation(threading.Thread):
    def __init__(self, input_queue, num_simulations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_queue = input_queue
        self.num_simulations = num_simulations
        self.jobs = {}

    def run(self):
        while True:
            message = self.command_queue.get()
            if message == DEATH:
                for name, (job, q) in self.jobs.items():
                    q.put(DEATH)
                    job.join()
                break
            message_queue = queue.Queue()
            j = MATLABScript(message_queue,
                             (float(message['iter']),
                              float(message['zmax']),
                              float(message['tmax']),
                              matlab.double(message['domain'].tolist()),
                              matlab.double(message['area'].tolist())))
            j.start()
            name = DIRSTR.format(
                    message['tmax'],
                    message['zmax'],
                    message['iter'])
            self.jobs[name] = (j, message_queue)


class MATLABScript(threading.Thread):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', metavar='d', type=str, nargs='*',
                        default='.',
                        help='Directory to watch for simulation data (recursive)')
    parser.add_argument('-n', '--num-workers', type=int, default=4,
                        help='Number of parse workers to use.')
    parser.add_argument('-s', '--num-simulations', type=int, default=2,
                        help='How many MATLAB sims to run')
    parser.add_argument('-db', '--database', type=str, default='ouroboros.db',
                        help='Database file to use.')
    args = parser.parse_args()
    if isinstance(args.directory, list):
        args.directory = args.directory[0]
    return args


if __name__ == '__main__':
    sys.exit(main())
