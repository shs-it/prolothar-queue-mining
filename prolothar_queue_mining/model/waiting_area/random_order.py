from random import Random

from prolothar_queue_mining.model.waiting_area.priority_queue import PriorityQueue
from prolothar_queue_mining.model.job import Job

class RandomOrderWaitingArea(PriorityQueue):
    """
    waiting area where jobs are served in completely random order
    """

    def __init__(self, seed: int = None):
        super().__init__()
        self.__random = Random(seed)

    def _compute_priority(self, arrival_time: int, job: Job) -> float:
        return self.__random.random()

    def get_discipline_name(self) -> str:
        return 'SIRO'

    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return exit_time

    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return -exit_time
