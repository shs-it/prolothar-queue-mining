import unittest

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.prediction.departure_time import FixedSojournTimeDepartureTimePredictor

class TestFixedSojournTimeDepartureTimePredictor(unittest.TestCase):

    def test_predict(self):
        expected_departure_times = {
            Job('A'): 15,
            Job('B'): 17,
            Job('C'): 42,
        }
        predictor = FixedSojournTimeDepartureTimePredictor(5)
        predicted_departure_times = predictor.predict([
            (
                Job('A'), 10
            ),
            (
                Job('B'), 12
            ),
            (
                Job('C'), 37
            )
        ])
        self.assertDictEqual(expected_departure_times, predicted_departure_times)

if __name__ == '__main__':
    unittest.main()
