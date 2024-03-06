from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.nr_of_servers.nr_of_servers_estimator import NrOfServersEstimator
from prolothar_queue_mining.inference.queue.nr_of_servers.utils import create_job_departure_order_list

class DummyNrOfServersEstimator(NrOfServersEstimator):
    """
    dummy estimator that outputs a fixed value
    """
    def __init__(self, nr_of_servers: int):
        self.__nr_of_servers = nr_of_servers

    def estimate_nr_of_servers(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> int:
        return self.__nr_of_servers
