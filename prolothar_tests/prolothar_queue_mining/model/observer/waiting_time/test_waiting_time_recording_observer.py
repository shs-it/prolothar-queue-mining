import unittest

from prolothar_queue_mining.model.observer.waiting_time import WaitingTimeRecordingObserver
from prolothar_queue_mining.model.job import Job

class TestWaitingTimeRecordingObserver(unittest.TestCase):

    def test_add_and_pop_jobs(self):
        observer = WaitingTimeRecordingObserver()
        observer.notify(Job('A'), 3, 8)
        observer.notify(Job('B'), 4, 13)
        observer.notify(Job('C'), 7, 16)
        observer.notify(Job('D'), 7, 21)
        observer.notify(Job('E'), 10, 35)

        self.assertEqual(25, observer.get_max_waiting_time())
        self.assertEqual(12.4, observer.get_mean_waiting_time())
        self.assertEqual(9, observer.get_median_waiting_time())
        self.assertEqual(([3,4,7,7,10],[5,9,9,14,25]), observer.get_timeseries_data())

if __name__ == '__main__':
    unittest.main()