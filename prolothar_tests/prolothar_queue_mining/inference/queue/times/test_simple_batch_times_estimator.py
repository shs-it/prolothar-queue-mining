import unittest

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.waiting_area import FirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import LastComeFirstServeWaitingArea
from prolothar_queue_mining.inference.queue.times import infer_service_times_batch

class TestSimpleBatchTimesEstimator(unittest.TestCase):

    def test_infer_service_times_batch_fcfs(self):
        observed_arrivals = [
            (Job('A'), 3),
            (Job('B'), 4),
            (Job('C'), 5),
            (Job('D'), 6),
            (Job('E'), 7),
            (Job('F'), 8),
            (Job('G'), 9),
        ]
        observed_departues = [
            (Job('A'), 5),
            (Job('B'), 5),
            (Job('C'), 8),
            (Job('D'), 8),
            (Job('E'), 12),
            (Job('F'), 12),
            (Job('G'), 12),
        ]
        _, batches, batch_service_times = infer_service_times_batch(
            observed_arrivals, observed_departues, FirstComeFirstServeWaitingArea(), 1)
        self.assertEqual(3, len(batches))
        expected_batches = [[Job('A'),Job('B')],[Job('C'),Job('D')],[Job('E'),Job('F'),Job('G')]]
        for actual_batch, expected_batch in zip(batches, expected_batches):
            self.assertCountEqual(actual_batch, expected_batch)
        self.assertEqual([1, 2, 3], batch_service_times)

if __name__ == '__main__':
    unittest.main()