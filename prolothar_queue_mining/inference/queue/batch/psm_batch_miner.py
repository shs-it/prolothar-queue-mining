from prolothar_queue_mining.model.job import Job

class PerformanceSpectrumBatchMiner():
    """
    https://github.com/multi-dimensional-process-mining/psm-batchmining/blob/master/src/main/java/BatchMiner.java
    """

    def __init__(self, max_delay: float = 0):
        self.__max_delay = max_delay

    def group_batches(
            self, observed_arrivals: list[tuple[Job, float]],
            observed_departures: list[tuple[Job, float]]) -> list[set[Job]]:
        departure_job_table = {job: departure_time for job,departure_time in observed_departures}
        all_observations = []
        for job, arrival_time in observed_arrivals:
            if job in departure_job_table:
                all_observations.append((job, arrival_time, departure_job_table[job]))
        all_observations.sort(key=lambda x: (x[2], x[1]))

        batch_list = []
        current_batch = set()
        for i in range(1, len(all_observations)):
            current_job, current_arrival_time, current_departure_time = all_observations[i]
            previous_job, previous_arrival_time, previous_departure_time = all_observations[i-1]
            current_batch.add(previous_job)
            if current_departure_time <= self.__max_delay + previous_departure_time \
            and current_arrival_time >= previous_arrival_time:
                current_batch.add(current_job)
            else:
                batch_list.append(current_batch)
                current_batch = set([current_job])
        if current_batch:
            batch_list.append(current_batch)
        return batch_list
