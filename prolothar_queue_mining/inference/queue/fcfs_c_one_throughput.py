from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import ServiceTimeWithDistribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.inference.queue.queue_miner import QueueMiner
from prolothar_queue_mining.inference.queue.utils import count_nr_of_jobs_in_system

class FcfsCOneThroughput(QueueMiner):
    """
    infers a queue with D=FCFS and c=1 and S=Degenerate(timesteps / number of jobs).
    only with at least 1 job in the system are considered to filter-out phases
    without the possibility to serve jobs.
    """

    def infer_queue(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> Queue:
        _, nr_of_jobs_in_system = count_nr_of_jobs_in_system(
            dict(observed_arrivals), dict(observed_departures))
        return Queue(
            None,
            [
                Server(ServiceTimeWithDistribution(DiscreteDegenerateDistribution(
                    round(
                        sum(
                            1 for n in nr_of_jobs_in_system if n > 0
                        ) / len(observed_departures)
                    )
                )))
            ]
        )
