import sys
from random import Random
from collections import defaultdict
from tqdm import trange

from prolothar_common.experiments.statistics import Statistics

from prolothar_queue_mining.prediction.departure_time.departure_time_predictor import DepartureTimePredictor

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.population import ListPopulation
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.model.observer.waiting_time import WaitingTimeRecordingObserver
from prolothar_queue_mining.model.environment import Environment

class QueueDepartureTimePredictor(DepartureTimePredictor):
    """
    predictor that uses a queue model to predict departure times
    """
    def __init__(self, queue: Queue, repetitions: int=1000, seed: int = None,
                 verbose: bool = False, mode: str = 'stochastic'):
        match mode:
            case 'stochastic':
                self.__queue = queue
            case 'mean':
                self.__queue = queue.copy_mean()
            case _:
                raise NotImplementedError(f'unsupported mode "{mode}')
        if (self.__queue.get_servers()[0].get_service_time_definition().is_deterministic()
             and self.__queue.get_batch_size_distribution().is_deterministic()):
            self.__repetitions = 1
        else:
            self.__repetitions = repetitions
        self.__seed_generator = None if seed is None else Random(seed)
        self.__verbose = verbose

    def predict_waiting_and_departure_times_distribution(
        self, arrivals: list[tuple[Job, int]]) -> tuple[dict[Job, list[int]], dict[Job, list[int]]]:
        arrival_process = FixedArrival(
            ListPopulation([job for job,_ in arrivals]),
            [time for _,time in arrivals]
        )
        predicted_waiting_times_collector = defaultdict(list)
        predicted_departure_times_collector = defaultdict(list)
        for _ in trange(self.__repetitions, disable=not self.__verbose):
            predicted_waiting_times, predicted_departure_times = self.__do_single_run(arrivals, arrival_process)
            for job, time in predicted_waiting_times.items():
                predicted_waiting_times_collector[job].append(time)
            for job, time in predicted_departure_times.items():
                predicted_departure_times_collector[job].append(time)
        return predicted_waiting_times_collector, predicted_departure_times_collector

    def __do_single_run(self, arrivals, arrival_process):
        queue = self.__queue.copy()
        queue.set_arrival_process(arrival_process.copy())
        queue.set_exit(ListCollectorExit())
        queue.set_waiting_time_observer(WaitingTimeRecordingObserver())
        queue.set_seed(None if self.__seed_generator is None else self.__seed_generator.randint(0, sys.maxsize))
        environment = Environment(verbose=False)
        queue.schedule_next_arrival(environment)
        environment.run_until_event_queue_is_empty()
        predicted_waiting_times = self.__extract_waiting_times_from_queue(arrivals, queue)
        predicted_departure_times = self.__extract_departure_times_from_queue(arrivals, queue)
        return predicted_waiting_times, predicted_departure_times

    def __extract_departure_times_from_queue(self, arrivals, queue):
        predicted_departure_times = dict(zip(*queue.get_exit().get_recording()))
        #if queue gets stuck because of missing batch elements, the best
        #prediction we can offer is the average sojourn time
        if len(queue.get_waiting_area()) > 0:
            average_sojourn_time = Statistics(
                    predicted_departure_times[job] - arrival_time
                    for job, arrival_time in arrivals if job in predicted_departure_times).mean()
            for job,arrival_time in arrivals:
                if job not in predicted_departure_times:
                    predicted_departure_times[job] = arrival_time + average_sojourn_time
        return predicted_departure_times

    def __extract_waiting_times_from_queue(self, arrivals, queue):
        predicted_waiting_times = queue.get_waiting_time_observer().get_waiting_time_per_job_dict()
        #if queue gets stuck because of missing batch elements, the best
        #prediction we can offer is the average waiting time
        average_predicted_waiting_time = queue.get_waiting_time_observer().get_mean_waiting_time()
        for job,_ in arrivals:
            if job not in predicted_waiting_times:
                predicted_waiting_times[job] = average_predicted_waiting_time
        return predicted_waiting_times