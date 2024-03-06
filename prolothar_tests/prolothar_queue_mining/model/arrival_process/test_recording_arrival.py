import unittest

from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.arrival_process import RecordingArrival
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.population import ListPopulation

class TestRecordingArrival(unittest.TestCase):

    def test_get_next_job(self):
        arrival = RecordingArrival(FixedArrival(
            ListPopulation([Job('A'), Job('B'), Job('C')]),
            [10, 25, 42]
        ))
        arrival.get_next_job()
        arrival.get_next_job()
        self.assertEqual([10, 25], arrival.get_recorded_arrival_times())
        self.assertEqual([Job('A'), Job('B')], arrival.get_recorded_jobs())
        self.assertAlmostEqual(1/16, arrival.get_mean_arrival_rate(), delta=0.0001)

if __name__ == '__main__':
    unittest.main()