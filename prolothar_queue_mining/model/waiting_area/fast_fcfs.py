from typing import Iterator
from collections import deque

from prolothar_queue_mining.model.waiting_area.waiting_area import WaitingArea
from prolothar_queue_mining.model.job import Job

class FastFirstComeFirstServeWaitingArea(WaitingArea):
    """
    faster implementation of FirstComeFirstServeWaitingArea that does not look
    at the time stamps but just uses a queue for its internal storage of jobs
    """

    def __init__(self):
        self.__queue = deque()

    def add_job(self, arrival_time: int, job: Job):
        self.__queue.append(job)

    def has_next_job(self) -> bool:
        return bool(self.__queue)

    def pop_next_job(self, nr_of_jobs_in_system: int) -> Job:
        if not self.__queue:
            raise StopIteration()
        return self.__queue.popleft()

    def pop_batch(self, batch_size: int, nr_of_jobs_in_system: int) -> list[Job]:
        if len(self.__queue) < batch_size:
            raise StopIteration()
        return [self.__queue.popleft() for _ in range(batch_size)]

    def __len__(self):
        return len(self.__queue)

    def copy(self) -> WaitingArea:
        copy = FastFirstComeFirstServeWaitingArea()
        copy.__queue = deque(self.__queue)
        return copy

    def copy_empty(self) -> 'WaitingArea':
        return FastFirstComeFirstServeWaitingArea()

    def get_discipline_name(self) -> str:
        return 'FCFS'

    def any_order_iterator(self) -> Iterator[Job]:
        return iter(self.__queue)

    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return exit_time

    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return -exit_time
