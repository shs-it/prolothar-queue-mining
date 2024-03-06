import unittest

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.waiting_area import FirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import FastFirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.waiting_area import LastComeFirstServeWaitingArea
from prolothar_queue_mining.inference.queue.times import infer_waiting_and_service_times

class TestSimpleTimesEstimator(unittest.TestCase):

    def test_infer_waiting_and_service_times_fcfs(self):
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
        inferred_waiting_times, inferred_service_times, _ = infer_waiting_and_service_times(
            observed_arrivals, observed_departues, FirstComeFirstServeWaitingArea(), 1)
        self.assertDictEqual(
            {
                Job('A'): 0,
                Job('B'): 0,
                Job('C'): 2,
                Job('D'): 5,
                Job('E'): 5,
                Job('F'): 5
            },
            inferred_waiting_times
        )
        self.assertDictEqual(
            {
                Job('A'): 1,
                Job('B'): 3,
                Job('C'): 4,
                Job('D'): 1,
                Job('E'): 1,
                Job('F'): 1
            },
            inferred_service_times
        )

    def test_infer_waiting_and_service_times_lcfs(self):
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
            (Job('E'), 12),
            (Job('D'), 13),
            (Job('F'), 14),
        ]
        inferred_waiting_times, inferred_service_times, _ = infer_waiting_and_service_times(
            observed_arrivals, observed_departues, LastComeFirstServeWaitingArea(), 2)
        self.assertDictEqual(
            {
                Job('A'): 0,
                Job('B'): 0,
                Job('C'): 0,
                Job('D'): 6,
                Job('E'): 0,
                Job('F'): 3
            },
            inferred_waiting_times
        )
        self.assertDictEqual(
            {
                Job('A'): 1,
                Job('B'): 3,
                Job('C'): 6,
                Job('D'): 1,
                Job('E'): 5,
                Job('F'): 3
            },
            inferred_service_times
        )

    def test_infer_waiting_and_service_times_with_negative_service(self):

        observed_arrivals = [
            (Job('A'), 0),
            (Job('B'), 1),
            (Job('C'), 2),
        ]
        observed_departures = [
            (Job('B'), 2),
            (Job('A'), 3),
            (Job('C'), 4),
        ]

        waiting_time_per_job, service_time_per_job, jobs_per_server = infer_waiting_and_service_times(
            observed_arrivals, observed_departures,
            FastFirstComeFirstServeWaitingArea(), 1)

        self.assertDictEqual(
            {
                Job('A'): 0,
                Job('B'): 2,
                Job('C'): 1
            },
            waiting_time_per_job
        )
        self.assertDictEqual(
            {
                Job('A'): 3,
                Job('B'): -1,
                Job('C'): 1
            },
            service_time_per_job
        )
        self.assertEqual(
            [[Job('A'), Job('B'), Job('C')]],
            jobs_per_server
        )

if __name__ == '__main__':
    unittest.main()