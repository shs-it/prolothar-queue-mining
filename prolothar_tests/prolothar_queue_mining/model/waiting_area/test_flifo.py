import unittest

from prolothar_queue_mining.model.waiting_area import FlifoWaitingArea
from prolothar_queue_mining.model.job import Job

class TestFlifoWaitingArea(unittest.TestCase):

    def test_add_and_pop_jobs_fifo_on_low_load(self):
        waiting_area = FlifoWaitingArea(2, fifo_on_low_load=True)
        waiting_area.add_job(3, Job('A'))
        waiting_area.add_job(7, Job('B'))
        waiting_area.add_job(10, Job('C'))
        waiting_area.add_job(14, Job('D'))

        self.assertEqual(Job('D'), waiting_area.pop_next_job(4))
        self.assertEqual(Job('C'), waiting_area.pop_next_job(3))
        self.assertEqual(Job('A'), waiting_area.pop_next_job(2))
        self.assertEqual(Job('B'), waiting_area.pop_next_job(1))
        self.assertRaises(StopIteration, lambda: waiting_area.pop_next_job(1))

    def test_add_and_pop_jobs_fifo_on_high_load(self):
        waiting_area = FlifoWaitingArea(2, fifo_on_low_load=False)
        waiting_area.add_job(3, Job('A'))
        waiting_area.add_job(7, Job('B'))
        waiting_area.add_job(10, Job('C'))
        waiting_area.add_job(14, Job('D'))

        self.assertEqual(Job('A'), waiting_area.pop_next_job(4))
        self.assertEqual(Job('B'), waiting_area.pop_next_job(3))
        self.assertEqual(Job('D'), waiting_area.pop_next_job(2))
        self.assertEqual(Job('C'), waiting_area.pop_next_job(1))

if __name__ == '__main__':
    unittest.main()