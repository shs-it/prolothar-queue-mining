import numpy as np
from sklearn.tree import DecisionTreeClassifier

from prolothar_queue_mining.inference.queue.waiting_area.waiting_area_estimator import WaitingAreaEstimator
from prolothar_queue_mining.inference.queue.utils import count_nr_of_jobs_in_system

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import OracleServiceTime
from prolothar_queue_mining.model.observer.waiting_time import ServeOrderRecorder
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.waiting_area import FlifoWaitingArea
from prolothar_queue_mining.model.waiting_area import DepartureScheduledWaitingArea
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.environment import Environment

class FlifoWaitingAreaEstimator(WaitingAreaEstimator):

    def infer_waiting_area(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> WaitingArea|None:
        arrival_per_job = dict(observed_arrivals)
        departure_time_per_job = dict(observed_departures)
        serve_order_recorder = self.__run_imitative_queue(observed_arrivals, departure_time_per_job)
        a, nr_of_jobs_in_system_per_time = count_nr_of_jobs_in_system(
            arrival_per_job, departure_time_per_job)

        x_nr_of_jobs_in_system = []
        y_fifo_or_lifo = []
        for served_job, waiting_job_list in serve_order_recorder.get_recording():
            if not waiting_job_list:
                continue
            departure_time = departure_time_per_job[served_job]
            if departure_time < len(nr_of_jobs_in_system_per_time):
                min_arrival_time = min(map(arrival_per_job.__getitem__, waiting_job_list))
                max_arrival_time = max(map(arrival_per_job.__getitem__, waiting_job_list))
                arrival_time_of_served_job = arrival_per_job[served_job]
                if arrival_time_of_served_job <= min_arrival_time:
                    x_nr_of_jobs_in_system.append(
                        nr_of_jobs_in_system_per_time[departure_time]
                    )
                    y_fifo_or_lifo.append(0)
                if arrival_time_of_served_job >= max_arrival_time:
                    x_nr_of_jobs_in_system.append(
                        nr_of_jobs_in_system_per_time[departure_time]
                    )
                    y_fifo_or_lifo.append(1)
        if not y_fifo_or_lifo:
            return None
        tree = DecisionTreeClassifier(max_leaf_nodes=2, random_state=42)
        tree.fit(np.array(x_nr_of_jobs_in_system).reshape(-1, 1), np.array(y_fifo_or_lifo))
        threshold_list = [t for t in tree.tree_.threshold if t > 0]
        if len(threshold_list) == 1:
            left_class = int(np.argmax(tree.tree_.value[1]))
            right_class = int(np.argmax(tree.tree_.value[2]))
            if left_class != right_class:
                # +1 because nr_of_jobs_in_system does not account for the job itself
                return FlifoWaitingArea(
                    int(threshold_list[0])+1,
                    fifo_on_low_load=left_class < right_class
                )

    def __run_imitative_queue(
            self, observed_arrivals: list[tuple[Job, int]],
            departure_time_per_job: dict[Job, int]) -> ServeOrderRecorder:
        environment = Environment()
        oracle_waiting_area = DepartureScheduledWaitingArea(departure_time_per_job)
        serve_order_recorder = ServeOrderRecorder(oracle_waiting_area, departure_time_per_job)
        queue = Queue(
            FixedArrival.create_from_observation(observed_arrivals),
            [Server(OracleServiceTime(environment, departure_time_per_job))],
            waiting_area=oracle_waiting_area,
            waiting_time_observer=serve_order_recorder
        )
        queue.schedule_next_arrival(environment)
        environment.run_until_event_queue_is_empty()
        return serve_order_recorder

