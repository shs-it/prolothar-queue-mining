import numpy as np

from prolothar_queue_mining.prediction.departure_time.departure_time_predictor import DepartureTimePredictor

from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.ras import RecurrentPointProcessArrivalModel
from prolothar_queue_mining.inference.queue.ras import ServiceTimeModel

class RasDepartureTimePredictor(DepartureTimePredictor):
    """
    predictor that uses an ArrivalProcessModel and a ServiceTimeModel from
    prolothar_queue_mining.inference_queue.ras package
    """
    def __init__(
            self, arrival_process_model: RecurrentPointProcessArrivalModel,
            service_time_model: ServiceTimeModel):
        self.__arrival_process_model = arrival_process_model
        self.__service_time_model = service_time_model

    def predict_waiting_and_departure_times_distribution(
            self, arrivals: list[tuple[Job, int]]) -> tuple[None, dict[Job, list[int]]]:
        hidden_states = self.__arrival_process_model.compute_hidden_states(arrivals)
        jobs = [job for job,_ in arrivals]
        predicted_service_times = self.__service_time_model.sample(hidden_states, jobs)[0,:,:]
        average_predicted_service_time_row = predicted_service_times.mean(axis=0)
        #if test set is quite small, there can be a mismatch in the batch sizes such that
        #some jobs cannot be predicted, because there is no computed hidded state for them
        for _ in range(predicted_service_times.shape[0], len(jobs)):
            predicted_service_times = np.vstack((predicted_service_times, average_predicted_service_time_row))
        predicted_departure_times_collector = {
            job: (arrivals[i][1] + predicted_service_times[i,:]).tolist()
            for i, job in enumerate(jobs)
        }
        return None, predicted_departure_times_collector

