from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.environment import Environment

from prolothar_queue_mining.inference.queue.nr_of_servers.nr_of_servers_estimator import NrOfServersEstimator

class LowerBoundEstimator(NrOfServersEstimator):
    """
    lower bound estimator for queues with a given queuing descipline. the estimate
    is given by the minimal number of servers to explain the departure order.
    """

    def __init__(self, waiting_area: WaitingArea):
        self.__waiting_area = waiting_area

    def estimate_nr_of_servers(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> int:
        #we need data to make sensible inference
        if not observed_arrivals or not observed_departures:
            return 1

        exit_time_per_job = {job: exit_time for job, exit_time in observed_departures}
        observed_arrivals = sorted(
            ((job, time) for job, time in observed_arrivals if job in exit_time_per_job),
            key=lambda job_and_time: (
                job_and_time[1],
                self.__waiting_area.get_best_case_sort_key_for_synchronized_arrival(
                    job_and_time[0], exit_time_per_job[job_and_time[0]]
                )
            )
        )
        #reconstruction of order does only make sense for jobs for which we
        #have observed arrival
        arrived_jobs = set([job for job,_ in observed_arrivals])
        departed_jobs = [job for job,_ in observed_departures if job in arrived_jobs][::-1]

        jobs_at_service = set()
        arrived_jobs = set()
        waiting_area = self.__waiting_area.copy()
        observed_arrivals = observed_arrivals[::-1]
        nr_of_servers = 1
        while departed_jobs:
            next_leaving_job = departed_jobs.pop()
            while next_leaving_job not in arrived_jobs:
                arriving_job, arrival_time = observed_arrivals.pop()
                waiting_area.add_job(arrival_time, arriving_job)
                arrived_jobs.add(arriving_job)
            try:
                jobs_at_service.remove(next_leaving_job)
            except KeyError:
                while True:
                    next_served_job = waiting_area.pop_next_job(len(waiting_area) + len(jobs_at_service))
                    nr_of_jobs_in_service_with_different_exit_time = 0
                    exit_time = exit_time_per_job[next_leaving_job]
                    for job_in_service in jobs_at_service:
                        if exit_time != exit_time_per_job[job_in_service]:
                            nr_of_jobs_in_service_with_different_exit_time += 1
                    nr_of_servers = max(nr_of_servers, nr_of_jobs_in_service_with_different_exit_time + 1)
                    if next_served_job == next_leaving_job:
                        break
                    else:
                        jobs_at_service.add(next_served_job)

        return nr_of_servers
