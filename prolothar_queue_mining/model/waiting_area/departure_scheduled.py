from prolothar_queue_mining.model.waiting_area.priority_queue import PriorityQueue
from prolothar_queue_mining.model.job import Job

class DepartureScheduledWaitingArea(PriorityQueue):
    """
    waiting area where the jobs are popped according to their (scheduled)
    departure time
    """

    def __init__(self, scheduled_departure_times: dict[Job, int]):
        """
        creates a new DepartureScheduledWaitingArea

        Parameters
        ----------
        scheduled_departure_times : dict[Job, int]
            assigns departure times to jobs. jobs are server in increasing order
            of departure time. if a job has no entry in this dictionary, it
            will never be served
        """
        super().__init__()
        self.__scheduled_departure_times = scheduled_departure_times

    def has_next_job(self):
        return super().has_next_job() and next(self.any_order_iterator()) in self.__scheduled_departure_times

    def _compute_priority(self, arrival_time: int, job: Job) -> float:
        return self.__scheduled_departure_times.get(job, float('inf'))

    def get_discipline_name(self) -> str:
        return 'DepartureScheduled(...)'

    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return exit_time

    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return -exit_time

    def copy(self) -> 'DepartureScheduledWaitingArea':
        if len(self) != 0:
            raise NotImplementedError()
        return DepartureScheduledWaitingArea(self.__scheduled_departure_times)

    def copy_empty(self) -> 'DepartureScheduledWaitingArea':
        return DepartureScheduledWaitingArea(self.__scheduled_departure_times)