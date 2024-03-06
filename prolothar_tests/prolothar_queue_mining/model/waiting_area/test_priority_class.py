import unittest

from prolothar_queue_mining.model.waiting_area import FirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import LastComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import PriorityClassWaitingArea
from prolothar_queue_mining.model.job import Job

class TestPriorityClassWaitingArea(unittest.TestCase):

    def test_add_and_pop_jobs_fifo(self):
        waiting_area = PriorityClassWaitingArea('prio', ['A', 'B', 'C'], FirstComeFirstServeWaitingArea)
        waiting_area.add_job(3, Job('1', {'prio': 'A'}))
        waiting_area.add_job(7, Job('2', {'prio': 'B'}))
        waiting_area.add_job(10, Job('3', {'prio': 'C'}))
        waiting_area.add_job(14, Job('4', {'prio': 'A'}))

        self.assertEqual('1', waiting_area.pop_next_job(4).job_id)
        self.assertEqual('4', waiting_area.pop_next_job(3).job_id)
        self.assertEqual('2', waiting_area.pop_next_job(2).job_id)
        self.assertEqual('3', waiting_area.pop_next_job(1).job_id)

    def test_add_and_pop_jobs_lifo(self):
        waiting_area = PriorityClassWaitingArea('prio', ['A', 'B', 'C'], LastComeFirstServeWaitingArea)
        waiting_area.add_job(3, Job('1', {'prio': 'A'}))
        waiting_area.add_job(7, Job('2', {'prio': 'B'}))
        waiting_area.add_job(10, Job('3', {'prio': 'C'}))
        waiting_area.add_job(14, Job('4', {'prio': 'A'}))

        self.assertEqual('4', waiting_area.pop_next_job(4).job_id)
        self.assertEqual('1', waiting_area.pop_next_job(3).job_id)
        self.assertEqual('2', waiting_area.pop_next_job(2).job_id)
        self.assertEqual('3', waiting_area.pop_next_job(1).job_id)

if __name__ == '__main__':
    unittest.main()