from prolothar_queue_mining.model.observer.queue_length.queue_length_observer import QueueLengthObserver

class NullQueueLengthObserver(QueueLengthObserver):
    """
    a dummy observer for the queue length, that does nothing (null object pattern)
    """

    def notify(self, current_time: float, queue_length: int):
        #as the name suggests: do nothing
        pass