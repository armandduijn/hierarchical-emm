import time
from emm import config
from queue import Queue
import pandas as pd
from emm.queue import MinPriorityQueue, Result
from emm.description import refine, description_factory
from emm.measures import QualityMeasure


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed_time(self) -> float:
        if self.start_time is None:
            raise RuntimeError('Call start() before retrieving the elapsed time.')

        return time.time() - self.start_time


def satisfies_all(description, coverage, quality):
    if coverage < config.MIN_COVERAGE or coverage > config.MAX_COVERAGE:
        return False

    if quality < config.MIN_QUALITY:
        return False

    if len(description) < config.MIN_DESCRIPTION_LENGTH:
        return False

    return True


def get_initial_seed(data: pd.DataFrame) -> Result:
    return Result(quality=-1, description=description_factory(conditions=[], data=data))


def beam_search(data, targets, quality_measure: QualityMeasure, options = {}):
    timer = Timer()
    timer.start()  # Keep track of execution time for debugging

    set_options(options)

    candidate_queue = Queue(maxsize=0)
    candidate_queue.put(get_initial_seed(data))

    result_set = MinPriorityQueue(max_size=config.RESULT_SET_SIZE)

    # Print settings
    print('Settings:')
    for var in [x for x in dir(config) if not x.startswith('__')]:
        print(f'  {var}={getattr(config, var)}')

    print('Setting-up quality measure...')
    quality_measure.set_data(data)

    print('Finding subgroups...')

    for depth in range(0, config.SEARCH_DEPTH):
        beam = MinPriorityQueue(max_size=config.BEAM_WIDTH)

        while not candidate_queue.empty():
            seed = candidate_queue.get().description

            for description in refine(data, targets, seed):
                coverage, quality = quality_measure.calculate(description)

                # Check if the description satisfies the constraints
                if not satisfies_all(description=description, coverage=coverage, quality=quality):
                    continue  # Continue with next candidate description

                result = Result(quality=quality, description=description)

                # Check if the description is novel
                if not result_set.contains(result):
                    result_set.put(result)

                beam.put(result)

        print(f'Best subgroups at depth {depth}:')
        while not beam.empty():
            candidate = beam.get()  # Not sure about get(), pseudo code uses get_front_element()

            print(f'quality = {round(candidate.quality, 5)}, description = {candidate.description.to_querystring()}')

            candidate_queue.put(candidate)

    top_q = list(result_set)
    top_q.reverse()  # Sort by descending quality

    print('Done.')
    print(f'Finished in {round(timer.elapsed_time())} seconds.')

    return top_q


def set_options(options):
    # TODO: Change config.py to allow all constants to be updated dynamically

    if 'SEARCH_DEPTH' in options:
        config.SEARCH_DEPTH = options['SEARCH_DEPTH']

    if 'BEAM_WIDTH' in options:
        config.BEAM_WIDTH = options['BEAM_WIDTH']

    if 'MAX_COVERAGE' in options:
        config.MAX_COVERAGE = options['MAX_COVERAGE']