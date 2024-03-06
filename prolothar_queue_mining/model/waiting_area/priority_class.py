from typing import Callable, Iterator
from math import log2

import prolothar_common.mdl_utils as mdl_utils

from prolothar_queue_mining.model.waiting_area.waiting_area import WaitingArea
from prolothar_queue_mining.model.job import Job

class PriorityClassWaitingArea(WaitingArea):
    """
    Waiting area with priority classes. This waiting area consists of sub-waiting
    area, one for each priority class. This enables FIFO or LIFO or any other handling of
    jobs with the same priority
    """

    def __init__(
        self, priority_feature_name: str, priority_classes: list,
        sub_waiting_area_factory: Callable[[], WaitingArea]):
        """
        Parameters
        ----------
        priority_feature_name : str
            name of the feature that contains the priority class for a job
        priority_classes : list
            order of the priority classes. jobs with the first priority class
            in the list will be served first
        sub_waiting_area_factory : Callable[[], WaitingArea]
            used to create the sub-waiting areas for each priority class.
            e.g. FirstComeFirstServeWaitingArea would enable FIFO serving for
            jobs with the same priority class
        """
        self.__priority_feature_name = priority_feature_name
        self.__priority_classes = priority_classes
        self.__priority_class_to_index = {
            priority_class: i for i,priority_class in enumerate(priority_classes)
        }
        self.__sub_waiting_area_factory = sub_waiting_area_factory
        self.__sub_waiting_areas = [sub_waiting_area_factory() for _ in priority_classes]

    def add_job(self, arrival_time: int, job: Job):
        """
        adds a job to this waiting area
        """
        #unknown categories get lowest priority
        self.__sub_waiting_areas[
            self.__priority_class_to_index.get(
                job.features[self.__priority_feature_name],
                -1
            )
        ].add_job(arrival_time, job)

    def has_next_job(self) -> bool:
        """
        returns True if there is a waiting job, otherwise returns False
        """
        for waiting_area in self.__sub_waiting_areas:
            if waiting_area.has_next_job():
                return True
        return False

    def pop_next_job(self, nr_of_jobs_in_system: int) -> Job:
        """
        returns the next waiting job if there is one. otherwise raises a StopIteration.
        """
        for waiting_area in self.__sub_waiting_areas:
            if waiting_area.has_next_job():
                return waiting_area.pop_next_job(nr_of_jobs_in_system)
        raise StopIteration()

    def pop_batch(self, batch_size: int, nr_of_jobs_in_system: int) -> list[Job]:
        """
        returns the next "batch_size" waiting jobs if there are enough jobs waiting.
        otherwise raises a StopIteration.
        """
        if len(self) >= batch_size:
            batch: list[Job] = []
            for waiting_area in self.__sub_waiting_areas:
                batch.extend(waiting_area.pop_batch(
                    min(batch_size - len(batch), len(waiting_area)),
                    nr_of_jobs_in_system
                ))
                if len(batch) == batch_size:
                    return batch
        raise StopIteration()

    def __len__(self):
        nr_of_jobs = 0
        for waiting_area in self.__sub_waiting_areas:
            nr_of_jobs += len(waiting_area)
        return nr_of_jobs

    def copy(self) -> WaitingArea:
        if len(self) > 0:
            raise NotImplementedError()
        return self.copy_empty()

    def copy_empty(self) -> 'WaitingArea':
        return PriorityClassWaitingArea(
            self.__priority_feature_name,
            self.__priority_classes,
            self.__sub_waiting_area_factory)

    def get_discipline_name(self) -> str:
        return (
            f'PQ({self.__priority_feature_name},'
            f'[{",".join(str(c) for c in self.__priority_classes)}],'
            f'{self.__sub_waiting_areas[0].get_discipline_name()})'
        )

    def any_order_iterator(self) -> Iterator[Job]:
        for sub_waiting_area in self.__sub_waiting_areas:
            yield from sub_waiting_area.any_order_iterator()

    def get_mdl(self, nr_of_categorical_features: int) -> float:
        return log2(nr_of_categorical_features) + mdl_utils.sum_log_i_from_1_to_n(len(self.__priority_classes))

    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return (
            self.__priority_class_to_index.get(
                job.features[self.__priority_feature_name],
                len(self.__priority_classes)
            ),
            self.__sub_waiting_areas[0].get_best_case_sort_key_for_synchronized_arrival(job, exit_time)
        )

    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return (
            self.__priority_class_to_index.get(
                job.features[self.__priority_feature_name],
                len(self.__priority_classes)
            ),
            self.__sub_waiting_areas[0].get_worst_case_sort_key_for_synchronized_arrival(job, exit_time)
        )

