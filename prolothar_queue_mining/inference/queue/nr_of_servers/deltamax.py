from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.nr_of_servers.nr_of_servers_estimator import NrOfServersEstimator
from prolothar_queue_mining.inference.queue.nr_of_servers.utils import create_job_departure_order_list

class Deltamax(NrOfServersEstimator):
    """
    deltamax estimator for FCFS queues as described in

    Andrew Keith and Darryl Ahner and Raymond Hill
    "An order-based method for robust queue inference with stochastic arrival and departure times"
    Computers & Industrial Engineering
    2019

    originally implemented in
    https://github.com/ajkeith/UnobservableQueue.jl
    """

    def estimate_nr_of_servers(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> int:
        departure_order = create_job_departure_order_list(observed_arrivals, observed_departures)

        current_max = departure_order[0]
        deltamax = 0
        for job in departure_order[1:]:
            next_max = max(job, current_max)
            deltamax = max(deltamax, next_max - current_max)
            current_max = next_max
        return deltamax
