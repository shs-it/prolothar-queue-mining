import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from prolothar_queue_mining.inference.queue.waiting_area.waiting_area_estimator import WaitingAreaEstimator
from prolothar_queue_mining.inference.sklearn.job_to_vector_transformer_utils import create_job_to_vector_transformer_for_linear_model

from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.waiting_area import RegressorWaitingArea
from prolothar_queue_mining.model.waiting_area.regressor import LinearRegressor
from prolothar_queue_mining.model.event import Event
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.environment import Environment

class LinearRegressionEstimator(WaitingAreaEstimator):

    def __init__(
            self, numerical_feature_names: list[str], categorical_feature_names: list[str],
            max_nr_of_epochs: int = 1000, batch_size: int = 64, seed: int = None,
            verbose: bool = False, early_stopping_epsilon: float = 0.00001,
            early_stopping_patience: int = 50):
        self.__categorical_feature_names = categorical_feature_names
        self.__numerical_feature_names = numerical_feature_names
        self.__max_nr_of_epochs = max_nr_of_epochs
        self.__seed = seed
        self.__batch_size = batch_size
        self.__verbose = verbose
        self.__early_stopping_epsilon = early_stopping_epsilon
        self.__early_stopping_patience = early_stopping_patience

    def infer_waiting_area(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> WaitingArea:

        departure_time_per_job = dict(observed_departures)

        job_to_vector_transformer = create_job_to_vector_transformer_for_linear_model(
            departure_time_per_job.keys(), self.__categorical_feature_names,
            self.__numerical_feature_names)

        top_element_matrix, remaining_element_matrix = self.__create_feature_matrices(
            observed_arrivals, observed_departures,
            {
                job: np.hstack((job_to_vector_transformer.transform(job), arrival_time))
                for job, arrival_time in observed_arrivals
            }
        )

        number_of_samples = top_element_matrix.shape[0]
        dataset_iterator = self.__create_dataset_iterator(
            top_element_matrix, remaining_element_matrix, number_of_samples)

        weights = self.__find_best_weights(number_of_samples, dataset_iterator, top_element_matrix.shape[1])

        return RegressorWaitingArea(job_to_vector_transformer, LinearRegressor(weights))

    def __find_best_weights(self, number_of_samples, dataset_iterator, nr_of_features):
        weights = tf.Variable(tf.random.normal((1, nr_of_features)), name='weights')
        optimizer = tf.keras.optimizers.Adam()
        best_loss = float('inf')
        best_weights = None
        current_patience = 0
        for epoch in range(self.__max_nr_of_epochs):
            epoch_loss = self.__iterate_one_epoch(number_of_samples, dataset_iterator, weights, optimizer)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_weights = weights.numpy()
            if epoch_loss >= best_loss + self.__early_stopping_epsilon:
                current_patience += 1
                if current_patience > self.__early_stopping_patience:
                    if self.__verbose:
                        print('early stopping')
                    break
            else:
                current_patience = 0

            if self.__verbose:
                print(f'loss of epoch {epoch}: {epoch_loss}')
        return best_weights.flatten()

    def __iterate_one_epoch(self, number_of_samples, dataset_iterator, weights, optimizer) -> float:
        epoch_loss = []
        for _ in range(int(number_of_samples/self.__batch_size)):
            batch_top_elements, batch_remaining_elements = next(dataset_iterator)
            with tf.GradientTape() as tape:
                output = tf.subtract(
                        tf.linalg.matmul(weights, tf.transpose(batch_top_elements)),
                        tf.linalg.matmul(weights, tf.transpose(batch_remaining_elements))
                    )
                loss = tf.reduce_sum(tf.maximum(0, output + 1))
            gradients = tape.gradient(loss, [weights])
            optimizer.apply_gradients(zip(gradients, [weights]))
            epoch_loss.append(loss.numpy())
        epoch_loss = np.array(epoch_loss).mean()
        return epoch_loss

    def __create_dataset_iterator(self, top_element_matrix, remaining_element_matrix, number_of_samples):
        dataset = tf.data.Dataset.from_tensor_slices(( top_element_matrix , remaining_element_matrix ))
        dataset = dataset.shuffle(
            number_of_samples, seed=self.__seed
        ).repeat(self.__max_nr_of_epochs).batch(self.__batch_size)
        dataset_iterator = iter(dataset)
        return dataset_iterator

    def __create_feature_matrices(
            self, observed_arrivals, observed_departures,
            job_to_vector) -> tuple[EagerTensor, EagerTensor]:
        jobs_in_system = set()
        top_element_matrix = []
        remaining_element_matrix = []

        environment = Environment()
        for job, arrival_time in observed_arrivals:
            environment.schedule_event(ArrivalEvent(job, arrival_time, jobs_in_system))
        for job, departure_time in observed_departures:
            environment.schedule_event(DepartureEvent(
                job, departure_time, jobs_in_system, job_to_vector,
                top_element_matrix, remaining_element_matrix))
        environment.run_timesteps(observed_departures[-1][1])

        top_element_matrix = tf.constant(np.array(top_element_matrix), dtype=tf.float32)
        remaining_element_matrix = tf.constant(np.array(remaining_element_matrix), dtype=tf.float32)
        return top_element_matrix, remaining_element_matrix

class ArrivalEvent(Event):

    def __init__(self, job: Job, arrival_time: int, jobs_in_system: set[Job]):
        super().__init__(arrival_time)
        self.__job = job
        self.__jobs_in_system = jobs_in_system

    def execute(self, environment: Environment):
        self.__jobs_in_system.add(self.__job)

class DepartureEvent(Event):

    def __init__(
            self, job: Job, departure_time: int,
            jobs_in_system: set[Job], job_to_vector: dict[Job, np.ndarray],
            top_element_matrix: list[np.ndarray], remaining_element_matrix: list[np.ndarray]):
        super().__init__(departure_time)
        self.__job = job
        self.__job_to_vector = job_to_vector
        self.__jobs_in_system = jobs_in_system
        self.__top_element_matrix = top_element_matrix
        self.__remaining_element_matrix = remaining_element_matrix

    def execute(self, environment: Environment):
        if self.__job in self.__jobs_in_system:
            self.__jobs_in_system.remove(self.__job)
            top_element_vector = self.__job_to_vector[self.__job]
            for remaining_job in self.__jobs_in_system:
                self.__top_element_matrix.append(top_element_vector)
                self.__remaining_element_matrix.append(self.__job_to_vector[remaining_job])
