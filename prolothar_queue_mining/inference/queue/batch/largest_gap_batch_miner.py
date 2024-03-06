from itertools import pairwise
from statistics import mean
import numpy as np
import kmeans1d

from prolothar_queue_mining.model.job import Job

class LargestGapBatchMiner():
    """
    jobs with the approximate the same departure time are assumed to be in a batch.
    the departure time of the first and last job in a batch can differ up to a
    auto-discovered max_delay parameter, the inter departure times are first sorted
    and then max_delay refers to the largest inter departure times before the
    largest gap
    """

    def group_batches(
            self, observed_arrivals: list[tuple[Job, float]],
            observed_departures: list[tuple[Job, float]]) -> list[set[Job]]:
        inter_departure_times = sorted(
            departure_time_b - departure_time_a
            for departure_time_a, departure_time_b
            in pairwise(departure_time for _,departure_time in observed_departures)
        )
        clusters, centroids = kmeans1d.cluster(inter_departure_times, 2)
        small_number_cluster_id = 0 if centroids[0] < centroids[1] else 1
        max_delay = 0
        for cluster_id, delay in zip(clusters, inter_departure_times):
            if cluster_id == small_number_cluster_id:
                max_delay = delay
            else:
                break
        batches = []
        current_batch = [observed_departures[0][0]]
        current_departure_time = observed_departures[0][1]
        for job, departure_time in observed_departures[1:]:
            if departure_time - max_delay > current_departure_time:
                batches.append(current_batch)
                current_departure_time = departure_time
                current_batch = []
            current_batch.append(job)
        batches.append(current_batch)
        return batches
