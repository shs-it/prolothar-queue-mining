from prolothar_queue_mining.model.waiting_area.priority_queue import PriorityQueue
from prolothar_queue_mining.model.job import Job

class LastComeFirstServeWaitingArea(PriorityQueue):
    """
    LIFO processing
    """

    def _compute_priority(self, arrival_time: int, job: Job) -> float:
        return -arrival_time

    def get_discipline_name(self) -> str:
        return 'LCFS'

    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return -exit_time

    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return exit_time
