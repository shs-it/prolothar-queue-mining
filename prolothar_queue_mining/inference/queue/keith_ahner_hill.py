import numpy as np

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import FixedServiceTime
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import FastLastComeFirstServeWaitingArea

from prolothar_queue_mining.inference.queue.queue_miner import QueueMiner
from prolothar_queue_mining.inference.queue.nr_of_servers import NrOfServersEstimator
from prolothar_queue_mining.inference.queue.nr_of_servers import COrder, COrderLcfs
from prolothar_queue_mining.inference.queue.times import infer_waiting_and_service_times

class KeithAhnerHill(QueueMiner):
    """
    estimates the number of servers for FCFS and LCFS
    queues, then takes the discipline with the least estimated number of servers
    and estimates a mean service time

    Andrew Keith and Darryl Ahner and Raymond Hill
    "An order-based method for robust queue inference with stochastic arrival and departure times"
    Computers & Industrial Engineering
    2019

    originally implemented in
    https://github.com/ajkeith/UnobservableQueue.jl
    """

    def __init__(self, nr_of_servers_estimator_list: list[tuple[NrOfServersEstimator, WaitingArea]] = None):
        if not nr_of_servers_estimator_list:
            self.__nr_of_servers_estimator_list = [
                (COrder(), FastFirstComeFirstServeWaitingArea()),
                (COrderLcfs(), FastLastComeFirstServeWaitingArea())]
        else:
            self.__nr_of_servers_estimator_list = nr_of_servers_estimator_list

    def infer_queue(
            self, observed_arrivals: list[tuple[Job, float]],
            observed_departures: list[tuple[Job, float]]) -> Queue:
        min_waiting_area = None
        min_c = float('inf')
        for nr_of_servers_estimator, waiting_area in self.__nr_of_servers_estimator_list:
            c = nr_of_servers_estimator.estimate_nr_of_servers(observed_arrivals, observed_departures)
            if c < min_c:
                min_c = c
                min_waiting_area = waiting_area
        return self.__infer_queue(
            observed_arrivals, observed_departures,
            min_waiting_area, min_c)

    def __infer_queue(
            self, observed_arrivals: list[tuple[Job, float]],
            observed_departures: list[tuple[Job, float]],
            waiting_area: WaitingArea, c: int) -> Queue:
        _, service_times, _ = infer_waiting_and_service_times(
            observed_arrivals, observed_departures, waiting_area, c)
        mean_service_time = np.mean([s for s in service_times.values()])
        return Queue(
            None,
            [Server(FixedServiceTime(mean_service_time)) for _ in range(c)],
            waiting_area=waiting_area
        )
