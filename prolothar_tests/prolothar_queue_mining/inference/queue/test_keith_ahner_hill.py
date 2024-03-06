import unittest

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.inference.queue import KeithAhnerHill

class TestKeithAhnerHill(unittest.TestCase):

    def test_infer_queue(self):
        keith_ahner_hill = KeithAhnerHill()
        observed_arrivals = [
            (Job('A'), 3),
            (Job('B'), 4),
            (Job('C'), 5),
            (Job('D'), 6),
            (Job('E'), 7),
            (Job('F'), 8),
        ]
        observed_departues = [
            (Job('A'), 4),
            (Job('B'), 7),
            (Job('C'), 11),
            (Job('D'), 12),
            (Job('E'), 13),
            (Job('F'), 14),
        ]
        inferred_queue = keith_ahner_hill.infer_queue(observed_arrivals, observed_departues)
        self.assertIsInstance(inferred_queue, Queue)

if __name__ == '__main__':
    unittest.main()