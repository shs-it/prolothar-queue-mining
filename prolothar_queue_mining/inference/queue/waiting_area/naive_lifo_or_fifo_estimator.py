from collections import defaultdict
import numpy as np

from prolothar_queue_mining.inference.queue.waiting_area.waiting_area_estimator import WaitingAreaEstimator

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import OracleServiceTime
from prolothar_queue_mining.model.observer.waiting_time import ServeOrderRecorder
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import FastLastComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import DepartureScheduledWaitingArea
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.environment import Environment

class NaiveLifoOrFifoWaitingAreaEstimator(WaitingAreaEstimator):

    def infer_waiting_area(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> WaitingArea:
        serve_order_recorder = self.__run_imitative_queue(observed_arrivals, observed_departures)
        arrival_per_job = dict(observed_arrivals)
        nr_of_times_fifo = 0
        nr_of_times_lifo = 0
        for served_job, waiting_job_list in serve_order_recorder.get_recording():
            arrival_time_of_served_job = arrival_per_job[served_job]
            for waiting_job in waiting_job_list:
                arrival_time_of_waiting_job = arrival_per_job[waiting_job]
                if arrival_time_of_served_job < arrival_time_of_waiting_job:
                    nr_of_times_fifo += 1
                elif arrival_time_of_served_job > arrival_time_of_waiting_job:
                    nr_of_times_lifo += 1
        if nr_of_times_fifo >= nr_of_times_lifo:
            return FastFirstComeFirstServeWaitingArea()
        else:
            return FastLastComeFirstServeWaitingArea()

    def __run_imitative_queue(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> ServeOrderRecorder:
        departure_time_per_job = {job: departure_time for job, departure_time in observed_departures}
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
        environment.run_timesteps(observed_departures[-1][1])
        return serve_order_recorder

