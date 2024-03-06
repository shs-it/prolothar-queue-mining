from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.observer.waiting_time import WaitingTimeObserver

class NullWaitingTimeObserver(WaitingTimeObserver):
    """
    a waiting time observer that does nothing (null object pattern)
    """

    def notify(self, job: Job, arrival_time: int, start_of_service_time: int):
        #as the name suggests, do nothing
        pass