from typing import Iterator
from collections import deque

from prolothar_queue_mining.model.waiting_area.waiting_area import WaitingArea
from prolothar_queue_mining.model.job import Job

class FlifoWaitingArea(WaitingArea):
    """
    a waiting area that depending on the number of jobs in the system either serves
    FIFO or LIFO (low and high load mode).
    """

    def __init__(self, load_threshold: int, fifo_on_low_load: bool = True):
        """
        creates a new FlifoWaitingArea

        Parameters
        ----------
        load_threshold : int
            up to this number of jobs in the system, jobs are popped in order
            of the low load mode. if the number of jobs in the system is higher
            than this number, jobs are popped in order of the high load mode.

        fifo_on_low_load : bool, optional
            if True, then jobs are returned fifo if the number of jobs
            in the system is low and otherwise lifo.
            if False, jobs are returned lifo if the number of jobs
            in the system is low and otherwise fifo
            by default True
        """
        self.__queue = deque()
        self.__load_threshold = load_threshold
        self.__fifo_on_low_load = fifo_on_low_load

    def add_job(self, arrival_time: int, job: Job):
        self.__queue.append(job)

    def has_next_job(self) -> bool:
        return bool(self.__queue)

    def pop_next_job(self, nr_of_jobs_in_system: int) -> Job:
        try:
            if nr_of_jobs_in_system <= self.__load_threshold:
                if self.__fifo_on_low_load:
                    return self.__queue.popleft()
                else:
                    return self.__queue.pop()
            else:
                if self.__fifo_on_low_load:
                    return self.__queue.pop()
                else:
                    return self.__queue.popleft()
        except IndexError:
            raise StopIteration()

    def pop_batch(self, batch_size: int, nr_of_jobs_in_system: int) -> list[Job]:
        if len(self.__queue) < batch_size:
            raise StopIteration()
        if nr_of_jobs_in_system <= self.__load_threshold:
            if self.__fifo_on_low_load:
                return [self.__queue.popleft() for _ in range(batch_size)]
            else:
                return [self.__queue.pop() for _ in range(batch_size)]
        else:
            if self.__fifo_on_low_load:
                return [self.__queue.pop() for _ in range(batch_size)]
            else:
                return [self.__queue.popleft() for _ in range(batch_size)]

    def __len__(self):
        return len(self.__queue)

    def copy(self) -> WaitingArea:
        copy = FlifoWaitingArea(self.__load_threshold, fifo_on_low_load=self.__fifo_on_low_load)
        copy.__queue = deque(self.__queue)
        return copy

    def copy_empty(self) -> 'WaitingArea':
        return FlifoWaitingArea(self.__load_threshold, fifo_on_low_load=self.__fifo_on_low_load)

    def get_discipline_name(self) -> str:
        if self.__fifo_on_low_load:
            return f'FLIFO({self.__load_threshold},FIFO->LIFO)'
        else:
            return f'FLIFO({self.__load_threshold},LIFO->FIFO)'

    def any_order_iterator(self) -> Iterator[Job]:
        return iter(self.__queue)

    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return exit_time

    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return -exit_time
