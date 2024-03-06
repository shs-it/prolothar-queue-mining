import unittest

from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.inference.queue.nr_of_servers import COrder

class TestCOrder(unittest.TestCase):

    def test_estimate_nr_of_servers(self):
        estimated_nr_of_servers = COrder().estimate_nr_of_servers(
            [
                (Job('A'), 10),
                (Job('B'), 42),
                (Job('C'), 55),
                (Job('D'), 67),
                (Job('E'), 98)
            ],
            [
                (Job('A'), 15),
                (Job('B'), 47),
                (Job('C'), 60),
                (Job('D'), 72),
                (Job('E'), 103)
            ],
        )
        self.assertEqual(1, estimated_nr_of_servers)

if __name__ == '__main__':
    unittest.main()