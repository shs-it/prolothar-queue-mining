from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.observer.waiting_time import WaitingTimeObserver

class ServeOrderRecorder(WaitingTimeObserver):
    """
    an observer for that records which job is served first in a list of waiting
    jobs
    """

    def __init__(self, waiting_area: WaitingArea, departure_time_per_job):
        """
        creates a new ServeOrderRecorder

        Parameters
        ----------
        waiting_area : WaitingArea
            the waiting_area is used to get the remaining, currently waiting jobs
            in the queue
        """
        self.__waiting_area = waiting_area
        self.__recording: list[tuple[Job, tuple[Job]]] = []
        self.departure_time_per_job = departure_time_per_job

    def notify(self, job: Job, arrival_time: int, start_of_service_time: int):
        if job not in self.departure_time_per_job:
            raise NotImplementedError((job, [j in self.departure_time_per_job for j in self.__waiting_area.any_order_iterator()]))
        self.__recording.append((job, tuple(self.__waiting_area.any_order_iterator())))

    def get_recording(self) -> list[tuple[Job, tuple[Job]]]:
        """
        returns the recording, a list where each entry is a tuple consisting
        of a served job and a tuple the remaining waiting jobs (can be empty).
        """
        return self.__recording
