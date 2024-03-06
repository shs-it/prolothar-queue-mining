from prolothar_queue_mining.model.arrival_process.arrival_process import ArrivalProcess
from prolothar_queue_mining.model.job import Job

class NullArrival(ArrivalProcess):
    """
    a dummy arrival process, where no jobs arrive at all
    """

    def get_next_job(self) -> tuple[int,Job]:
        raise StopIteration()

    def get_mean_arrival_rate(self) -> float:
        return 0

    def copy(self) -> ArrivalProcess:
        return NullArrival()

    def set_seed(self, seed: int):
        #no action required
        pass
