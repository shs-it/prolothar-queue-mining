from collections import defaultdict
import numpy as np

from prolothar_queue_mining.inference.queue.waiting_area.waiting_area_estimator import WaitingAreaEstimator

from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.service_time import OracleServiceTime
from prolothar_queue_mining.model.observer.waiting_time import ServeOrderRecorder
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.waiting_area import PriorityClassWaitingArea
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import FastLastComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import DepartureScheduledWaitingArea
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.environment import Environment

class PriorityClassWaitingAreaEstimator(WaitingAreaEstimator):

    def __init__(self, categorical_feature_names: list[str], epsilon: float = 0.5):
        self.__categorical_feature_names = categorical_feature_names
        self.__epsilon = epsilon

    def infer_waiting_area(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> WaitingArea:
        serve_order_recorder = self.__run_imitative_queue(observed_arrivals, observed_departures)
        uninformed_counter, attribute_value_partial_order_count = \
            self.__create_attribute_value_partial_order_count(serve_order_recorder)
        selected_feature, order_of_categories = self.__select_feature_with_lowest_entropy(
            uninformed_counter, attribute_value_partial_order_count)
        #there was no feature with more than one category => no priority class waiting area inferrable
        if selected_feature is None:
            return None
        return PriorityClassWaitingArea(
            selected_feature, order_of_categories,
            self.__infer_fifo_or_lifo_subwaiting_area(
                selected_feature, serve_order_recorder, observed_arrivals)
        )

    def __select_feature_with_lowest_entropy(
            self, uninformed_counter, attribute_value_partial_order_count) -> tuple[str, list]:
        lowest_entropy = float('inf')
        selected_feature = None
        best_raw_entropy_matrix = None
        categories_of_best_feature = None
        for feature, partial_order_counter in attribute_value_partial_order_count.items():
            categories = sorted(uninformed_counter[feature].keys())
            if len(categories) > 1:
                joint_probabilities = self.__compute_joint_probabilities(partial_order_counter, categories)
                conditional_probabilities = self.__compute_conditional_probabilities(
                    partial_order_counter, categories, joint_probabilities)
                raw_entropy_matrix = -np.multiply(joint_probabilities, np.log(conditional_probabilities))
                informed_entropy = np.sum(raw_entropy_matrix)
                if informed_entropy < lowest_entropy:
                    selected_feature = feature
                    lowest_entropy = informed_entropy
                    best_raw_entropy_matrix = raw_entropy_matrix
                    categories_of_best_feature = categories
        #we had no features with more than one category
        if selected_feature is None:
            return None, None
        entropy_per_category = np.sum(best_raw_entropy_matrix, axis=1)
        order_of_categories = [categories_of_best_feature[i] for i in np.argsort(entropy_per_category)]
        return selected_feature, order_of_categories

    def __compute_conditional_probabilities(
            self, partial_order_counter, categories, joint_probabilities) -> np.array:
        marginal_waiting_probabilities = np.array([
            sum(
                partial_order_counter[served_category][waiting_category] + self.__epsilon
                for served_category in categories
            )
            for waiting_category in categories
        ])
        marginal_waiting_probabilities = marginal_waiting_probabilities / np.sum(marginal_waiting_probabilities)
        marginal_waiting_probabilities = np.transpose(
            np.vstack(
                [marginal_waiting_probabilities]*(len(marginal_waiting_probabilities)-1)
            )
        )
        return np.divide(joint_probabilities, marginal_waiting_probabilities)

    def __compute_joint_probabilities(self, partial_order_counter, categories) -> np.array:
        joint_probabilities = np.array([
            [
                partial_order_counter[category][different_category] + self.__epsilon
                for category in categories
                if category != different_category
            ] for different_category in categories
        ])
        return joint_probabilities / np.sum(joint_probabilities)

    def __run_imitative_queue(
            self, observed_arrivals: list[tuple[Job, int]],
            observed_departures: list[tuple[Job, int]]) -> ServeOrderRecorder:
        departure_time_per_job = {job: departure_time for job, departure_time in observed_departures}
        environment = Environment()
        oracle_waiting_area = DepartureScheduledWaitingArea(departure_time_per_job)
        serve_order_recorder = ServeOrderRecorder(oracle_waiting_area, departure_time_per_job)
        queue = Queue(
            FixedArrival.create_from_observation(observed_arrivals),
            [Server(OracleServiceTime(environment, departure_time_per_job))],
            waiting_area=oracle_waiting_area,
            waiting_time_observer=serve_order_recorder
        )
        queue.schedule_next_arrival(environment)
        environment.run_timesteps(observed_departures[-1][1])
        return serve_order_recorder

    def __create_attribute_value_partial_order_count(
            self, serve_order_recorder: ServeOrderRecorder) -> tuple[dict, dict]:
        attribute_value_partial_order_count = {
            feature_name: defaultdict(lambda: defaultdict(int))
            for feature_name in self.__categorical_feature_names
        }
        uninformed_counter = {
            feature_name: defaultdict(int)
            for feature_name in self.__categorical_feature_names
        }

        for categorical_feature in self.__categorical_feature_names:
            for served_job, waiting_job_list in serve_order_recorder.get_recording():
                category_of_served_job = served_job.features[categorical_feature]
                uninformed_counter[categorical_feature][category_of_served_job] += 1
                partial_order_counter = attribute_value_partial_order_count[
                    categorical_feature][category_of_served_job]
                for waiting_job in waiting_job_list:
                    category_of_waiting_job = waiting_job.features[categorical_feature]
                    if category_of_waiting_job != category_of_served_job:
                        partial_order_counter[category_of_waiting_job] += 1

        return uninformed_counter, attribute_value_partial_order_count

    def __infer_fifo_or_lifo_subwaiting_area(
            self, selected_feature: str, serve_order_recorder: ServeOrderRecorder,
            observed_arrivals: list[tuple[Job, int]]):
        arrival_per_job = {job: time for job, time in observed_arrivals}
        nr_of_times_fifo = 0
        nr_of_times_lifo = 0
        for served_job, waiting_job_list in serve_order_recorder.get_recording():
            category_of_served_job = served_job.features[selected_feature]
            arrival_time_of_served_job = arrival_per_job[served_job]
            for waiting_job in waiting_job_list:
                category_of_waiting_job = waiting_job.features[selected_feature]
                if category_of_waiting_job == category_of_served_job:
                    arrival_time_of_waiting_job = arrival_per_job[waiting_job]
                    if arrival_time_of_served_job < arrival_time_of_waiting_job:
                        nr_of_times_fifo += 1
                    elif arrival_time_of_served_job > arrival_time_of_waiting_job:
                        nr_of_times_lifo += 1

        if nr_of_times_fifo >= nr_of_times_lifo:
            return FastFirstComeFirstServeWaitingArea
        else:
            return FastLastComeFirstServeWaitingArea
