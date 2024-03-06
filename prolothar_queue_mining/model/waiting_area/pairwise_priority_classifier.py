from typing import Union, Iterator
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import ClassifierMixin

from prolothar_queue_mining.model.waiting_area.waiting_area import WaitingArea
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer
from prolothar_queue_mining.model.job.job_to_vector_transformer import Scaler
from prolothar_queue_mining.model.job.job_to_vector_transformer import NoScaler

TrainingRelations = list[tuple[Job, Job, int, bool]]

class PairwisePriorityClassifier(ABC):
    """
    interface of a pairwise priority classifier.
    the classifier decides for two jobs which one is higher prioritized
    """
    @abstractmethod
    def should_job_a_be_served_before_job_b(self, job_a: Job, job_b: Job, difference_in_arrival_time: int) -> bool:
        """
        returns True iff job_a should be served before job_b

        Parameters
        ----------
        job_a : Job
        job_b : Job
        difference_in_arrival_time : int
            arrival time of job_a - arrival time of job_b
            => how much younger is job_a than job_b
        """

    @abstractmethod
    def train(self, training_relations: TrainingRelations):
        """
        trains the classifier with a set of training relations

        Parameters
        ----------
        training_relations : TrainingRelations
            a list of relations, where each relation consists of two jobs,
            their difference in arrival time and a boolean flag whether job_a
            was served before job_b
        """
class SklearnPairwisePriorityClassifier(PairwisePriorityClassifier):
    """
    interface of a pairwise priority classifier.
    the classifier decides for two jobs which one is higher prioritized
    """
    def __init__(
        self, classifier: ClassifierMixin, job_to_vector_transformer: JobToVectorTransformer,
        scaler_for_arrival_time_difference: Scaler = NoScaler()):
        self.__classifier = classifier
        self.__job_to_vector_transformer = job_to_vector_transformer
        self.__scaler_for_arrival_time_difference = scaler_for_arrival_time_difference

    def should_job_a_be_served_before_job_b(
            self, job_a: Job, job_b: Job, difference_in_arrival_time: int) -> bool:
        self.__classifier.predict(np.hstack((
            self.__job_to_vector_transformer.transform(job_a),
            self.__job_to_vector_transformer.transform(job_b),
            [self.__scaler_for_arrival_time_difference.scale(difference_in_arrival_time)]
        )).reshape(1, -1))

    def train(self, training_relations: TrainingRelations):
        self.__classifier.fit(
            np.vstack([
                np.hstack([
                    self.__job_to_vector_transformer.transform(relation[0]),
                    self.__job_to_vector_transformer.transform(relation[1]),
                    [self.__scaler_for_arrival_time_difference.scale(relation[2])]
                ])
                for relation in training_relations
            ]),
            np.array([
                relation[3]
                for relation in training_relations
            ]).transpose()
        )

    def __repr__(self):
        try:
            return f'{self.__classifier}({self.__classifier.coef_}+{self.intercept_})'
        except AttributeError:
            return str(self.__classifier)

class Node:
    def __init__(self, job: Job, arrival_time: int, next_node: 'Node' = None):
        self.job = job
        self.next_node = next_node
        self.arrival_time = arrival_time

class PairwisePriorityClassifierWaitingArea(WaitingArea):
    """
    waiting area where a pairwise classifier decides on the order of the jobs.
    the classifier decides for two jobs which one is higher prioritized
    """

    def __init__(self, pairwise_priority_classifier: PairwisePriorityClassifier):
        self.__head_node: Union[Node, None] = None
        self.__pairwise_priority_classifier = pairwise_priority_classifier
        self.__length = 0
        self.__jobs_for_learning: dict[str, (int, Job)] = {}
        self.__training_relations: TrainingRelations = []

    def add_job(self, arrival_time: int, job: Job):
        """
        adds a job to this waiting area
        """
        if not self.__head_node:
            self.__head_node = Node(job, arrival_time)
        elif self.__pairwise_priority_classifier.should_job_a_be_served_before_job_b(
                job, self.__head_node.job, arrival_time - self.__head_node.arrival_time):
            self.__head_node = Node(job, arrival_time, next_node=self.__head_node)
        else:
            current_node = self.__head_node
            while True:
                if current_node.next_node is None:
                    current_node.next_node = Node(job, arrival_time)
                    break
                elif self.__pairwise_priority_classifier.should_job_a_be_served_before_job_b(
                        job, current_node.next_node.job, arrival_time - current_node.next_node.arrival_time):
                    current_node.next_node = Node(job, arrival_time, next_node=current_node.next_node)
                    break
                else:
                    current_node = current_node.next_node
        self.__length += 1

    def add_job_for_learning(self, arrival_time: int, job: Job):
        self.__jobs_for_learning[job.job_id] = (arrival_time, job)

    def has_next_job(self) -> bool:
        """
        returns True if there is a waiting job, otherwise returns False
        """
        return self.__head_node is not None

    def pop_next_job(self, nr_of_jobs_in_system: int) -> Job:
        """
        returns the next waiting job if there is one. otherwise raises a StopIteration.
        """
        if self.__head_node is None:
            raise StopIteration()
        next_job = self.__head_node.job
        self.__head_node = self.__head_node.next_node
        self.__length -= 1
        return next_job

    def pop_next_job_for_learning(self, job_id: str):
        #arrival might not have been observed
        if job_id in self.__jobs_for_learning:
            arrival_time, job = self.__jobs_for_learning.pop(job_id)
            for other_arrival_time, other_job in self.__jobs_for_learning.values():
                self.__training_relations.append((
                    job, other_job, arrival_time - other_arrival_time, True
                ))
                self.__training_relations.append((
                    other_job, job, other_arrival_time - arrival_time, False
                ))

    def learn_classifier(self):
        self.__pairwise_priority_classifier.train(self.__training_relations)

    def pop_batch(self, batch_size: int, nr_of_jobs_in_system: int) -> list[Job]:
        """
        returns the next "batch_size" waiting jobs if there are enough jobs waiting.
        otherwise raises a StopIteration.
        """
        if len(self) >= batch_size:
            return [self.pop_next_job(nr_of_jobs_in_system) for _ in range(batch_size)]
        raise StopIteration()

    def __len__(self):
        """
        returns the number of jobs in this waiting area
        """
        return self.__length

    def copy(self) -> 'WaitingArea':
        """
        returns a deep copy of this waiting area with the same state.
        if an attribute of this waiting area uses a random seed and the
        random seed is "None", this will result in different behavior during
        simulation. this is not a bug, but the reason for this method.
        otherwise use the "deepcopy" python module.
        """
        copy = PairwisePriorityClassifierWaitingArea(self.__pairwise_priority_classifier)
        current_node = self.__head_node
        while current_node is not None:
            copy.add_job(current_node.arrival_time, current_node.job)
            current_node = current_node.next_node
        return copy

    def copy_empty(self) -> 'WaitingArea':
        return PairwisePriorityClassifierWaitingArea(self.__pairwise_priority_classifier)

    def get_discipline_name(self) -> str:
        """
        returns the name of queue discipline, i.e. "D" in Kendal's notation,
        e.g. FCFS, LCFS, PQ, SIRO, ...
        """
        return f"PQ({self.__pairwise_priority_classifier})"

    def any_order_iterator(self) -> Iterator[Job]:
        current_node = self.__head_node
        while current_node is not None:
            yield current_node.job
            current_node = current_node.next_node

    def get_best_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return exit_time

    def get_worst_case_sort_key_for_synchronized_arrival(self, job: Job, exit_time: int):
        return -exit_time
