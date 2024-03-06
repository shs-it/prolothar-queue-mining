import unittest

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.prediction.departure_time import OracleDepartureTimePredictor

class TestOracleDepartureTimePredictor(unittest.TestCase):

    def test_predict(self):
        actual_departure_times = {
            Job('A'): 15,
            Job('B'): 27,
            Job('C'): 32,
        }
        predictor = OracleDepartureTimePredictor(actual_departure_times)
        predicted_departure_times = predictor.predict([
            (
                Job('A'), 10
            ),
            (
                Job('B'), 12
            ),
            (
                Job('C'), 17
            )
        ])
        self.assertDictEqual(actual_departure_times, predicted_departure_times)

if __name__ == '__main__':
    unittest.main()
