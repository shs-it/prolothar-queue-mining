import unittest

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.inference.queue.batch.psm_batch_miner import PerformanceSpectrumBatchMiner

class TestPsmBatchMiner(unittest.TestCase):

    def test_infer_queue(self):
        observed_arrivals = [
            (Job('A'), 3),
            (Job('B'), 4),
            (Job('C'), 5),
            (Job('D'), 6),
            (Job('E'), 7),
            (Job('F'), 8),
        ]
        observed_departures = [
            (Job('A'), 4),
            (Job('B'), 7),
            (Job('C'), 11),
            (Job('D'), 12),
            (Job('E'), 13),
            (Job('F'), 14),
        ]
        batches = PerformanceSpectrumBatchMiner().group_batches(observed_arrivals, observed_departures)
        self.assertEqual(6, len(batches))

        batches = PerformanceSpectrumBatchMiner(max_delay=1).group_batches(observed_arrivals, observed_departures)
        self.assertEqual(3, len(batches))

if __name__ == '__main__':
    unittest.main()