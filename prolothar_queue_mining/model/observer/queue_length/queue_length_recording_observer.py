from prolothar_queue_mining.model.observer.queue_length.queue_length_observer import QueueLengthObserver

class QueueLengthRecordingObserver(QueueLengthObserver):
    """
    an observer for (statistical) analysis of recorded queue length over time
    """

    def __init__(self, min_time: float = 0):
        self.__maximal_queue_length = 0
        self.__queue_length_per_time: dict[float, int] = {}
        self.__min_time = min_time

    def notify(self, current_time: float, queue_length: int):
        if current_time >= self.__min_time:
            self.__maximal_queue_length = max(queue_length, self.__maximal_queue_length)
            self.__queue_length_per_time[current_time] = queue_length

    def get_max_queue_length(self) -> int:
        """
        returns the maximal observed queue length
        """
        return self.__maximal_queue_length

    def get_timeseries_data(self) -> tuple[list[float], list[int]]:
        """
        the raw data how the timeseries data changed over time.

        returns (timesteps, queue_lengths) where timesteps and queue_lengths
        are lists.
        """
        timesteps = []
        queue_lengths = []
        for time, length in sorted(self.__queue_length_per_time.items()):
            timesteps.append(time)
            queue_lengths.append(length)
        return timesteps, queue_lengths
