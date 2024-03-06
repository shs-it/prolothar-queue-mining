from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.server import CountingServer
from prolothar_queue_mining.model.service_time import OracleServiceTime
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.population import ListPopulation
from prolothar_queue_mining.model.environment import Environment

from prolothar_queue_mining.inference.queue.nr_of_servers.nr_of_servers_estimator import NrOfServersEstimator

class UpperBoundEstimator(NrOfServersEstimator):
    """
    upper bound estimator for queues with a given queuing descipline. the upper
    bound is given by the smallest number of servers for which all incoming jobs
    can be served without waiting time. more servers would result in unused servers
    """

    def __init__(self, waiting_area: WaitingArea, max_upper_bound: int = 1000):
        self.__waiting_area = waiting_area
        self.__max_upper_bound = max_upper_bound

    def estimate_nr_of_servers(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> int:
        exit_time_per_job = {job: exit_time for job, exit_time in observed_departures}
        observed_arrivals = sorted(
            ((job, time) for job, time in observed_arrivals if job in exit_time_per_job),
            key=lambda job_and_time: (
                job_and_time[1],
                self.__waiting_area.get_worst_case_sort_key_for_synchronized_arrival(
                    job_and_time[0], exit_time_per_job[job_and_time[0]]
                )
            )
        )
        environment = Environment(verbose=False)
        oracle_service_time = OracleServiceTime(environment, exit_time_per_job)
        queue = Queue(
                FixedArrival(
                    ListPopulation([job for job, _ in observed_arrivals]),
                    [arrival_time for _, arrival_time in observed_arrivals]
                ),
                [CountingServer(oracle_service_time) for _ in range(self.__max_upper_bound)],
                waiting_area=self.__waiting_area.copy())
        queue.schedule_next_arrival(environment)
        environment.run_timesteps(observed_departures[-1][1])

        for i, server in enumerate(queue.get_servers()):
            if server.get_nr_of_served_jobs() == 0:
                return i
        return UpperBoundEstimator(
            self.__waiting_area,
            max_upper_bound=self.__max_upper_bound*2).estimate_nr_of_servers(
                observed_arrivals, observed_departures)
