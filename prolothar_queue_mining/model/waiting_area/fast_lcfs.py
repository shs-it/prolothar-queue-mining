from typing import Iterator

from prolothar_queue_mining.model.waiting_area.waiting_area import WaitingArea
from prolothar_queue_mining.model.job import Job

class FastLastComeFirstServeWaitingArea(WaitingArea):
    """
    faster implementation of LastComeFirstServeWaitingArea that does not look
    at the time stamps but just uses a stack for its internal storage of jobs
    """

    def __init__(self):
        self.__stack = []

    def add_job(self, arrival_time: int, job: Job):
        self.__stack.append(job)

    def has_next_job(self) -> bool:
        return bool(self.__stack)

    def pop_next_job(self, nr_of_jobs_in_system: int) -> Job:
        if not self.__stack:
            raise StopIteration()
        return self.__stack.pop()

    def pop_batch(self, batch_size: int, nr_of_jobs_in_system: int) -> list[Job]:
        if len(self.__stack) < batch_size:
            raise StopIteration()
        return [self.__stack.pop() for _ in range(batch_size)]

    def __len__(self):
        return len(self.__stack)

    def copy(self) -> WaitingArea:
        copy = FastLastComeFirstServeWaitingArea()
        copy.__stack = list(self.__stack)
        return copy

    def copy_empty(self) -> 'WaitingArea':
        return FastLastComeFirstServeWaitingArea()

    def get_discipline_name(self) -> str:
        return 'LCFS'

    def any_order_iterator(self) -> Iterator[Job]:
        return iter(self.__stack)

    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return -exit_time

    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return exit_time
