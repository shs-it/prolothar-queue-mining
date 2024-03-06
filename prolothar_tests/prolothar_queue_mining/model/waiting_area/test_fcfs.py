import unittest

from prolothar_queue_mining.model.waiting_area import FirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.job import Job

class TestFirstComeFirstServeWaitingArea(unittest.TestCase):

    def test_add_and_pop_jobs(self):
        waiting_area = FirstComeFirstServeWaitingArea()
        waiting_area.add_job(3, Job('A'))
        waiting_area.add_job(7, Job('B'))
        waiting_area.add_job(10, Job('C'))
        waiting_area.add_job(14, Job('D'))

        self.assertEqual(Job('A'), waiting_area.pop_next_job(4))
        self.assertEqual(Job('B'), waiting_area.pop_next_job(3))
        self.assertEqual(Job('C'), waiting_area.pop_next_job(2))
        self.assertEqual(Job('D'), waiting_area.pop_next_job(1))

if __name__ == '__main__':
    unittest.main()