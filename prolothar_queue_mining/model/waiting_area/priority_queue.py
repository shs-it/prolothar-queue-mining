from typing import Iterator
from abc import abstractmethod

import heapq

from prolothar_queue_mining.model.waiting_area.waiting_area import WaitingArea
from prolothar_queue_mining.model.job import Job

class QueueElement:
    """
    internally used element of a PriorityQueue
    """
    def __init__(self, arrival_time: int, job: Job, priority: float):
        self.arrival_time = arrival_time
        self.job = job
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

class PriorityQueue(WaitingArea):
    """
    waiting area where the element with highest priority (lowest numerical value)
    is selected first
    """

    def __init__(self):
        self.__queue = []

    def add_job(self, arrival_time: int, job: Job):
        heapq.heappush(self.__queue, QueueElement(
            arrival_time, job, self._compute_priority(arrival_time, job)))

    def has_next_job(self) -> bool:
        return bool(self.__queue)

    def pop_next_job(self, nr_of_jobs_in_system: int) -> Job:
        if not self.has_next_job():
            raise StopIteration()
        return heapq.heappop(self.__queue).job

    def pop_batch(self, batch_size: int, nr_of_jobs_in_system: int) -> list[Job]:
        if len(self.__queue) < batch_size:
            raise StopIteration()
        return [heapq.heappop(self.__queue).job for _ in range(batch_size)]

    def __len__(self):
        return len(self.__queue)

    @abstractmethod
    def _compute_priority(self, arrival_time: int, job: Job) -> float:
        """
        computes the priority of a given job depending on its arrival time

        Parameters
        ----------
        arrival_time : float
            the time when job joined the waiting area
        job : Job
            the job that needs to get prioritized

        Returns
        -------
        float
            the priority of the job. jobs with lower priority value are served
            first
        """

    def copy(self) -> WaitingArea:
        copy = type(self)()
        copy.__queue = list(self.__queue)
        return copy

    def copy_empty(self) -> 'WaitingArea':
        return type(self)()

    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return (self._compute_priority(0, job), exit_time)

    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return (self._compute_priority(0, job), -exit_time)

    def any_order_iterator(self) -> Iterator[Job]:
        for element in self.__queue:
            yield element.job
