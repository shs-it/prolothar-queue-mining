import unittest

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.inference.queue import FcfsCOneThroughput

class TestFcfsCOneThroughput(unittest.TestCase):

    def test_infer_queue(self):
        queue_inference = FcfsCOneThroughput()
        observed_arrivals = [
            (Job('A'), 0),
            (Job('B'), 1),
            (Job('C'), 2),
            (Job('D'), 3),
            (Job('E'), 4),
            (Job('F'), 5),
        ]
        observed_departues = [
            (Job('A'), 4),
            (Job('B'), 7),
            (Job('C'), 11),
            (Job('D'), 12),
            (Job('E'), 13),
            (Job('F'), 14),
        ]
        inferred_queue = queue_inference.infer_queue(observed_arrivals, observed_departues)
        self.assertIsInstance(inferred_queue, Queue)
        self.assertEqual(inferred_queue.get_service_time_name(), 'ServiceTime(DegenerateDistribution(2))')

if __name__ == '__main__':
    unittest.main()