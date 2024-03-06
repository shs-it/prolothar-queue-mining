import unittest

from random import Random
from sklearn.linear_model import LogisticRegression

from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.waiting_area import PairwisePriorityClassifierWaitingArea
from prolothar_queue_mining.model.waiting_area.pairwise_priority_classifier import SklearnPairwisePriorityClassifier
from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.job.job_to_vector_transformer import JobToVectorTransformer
from prolothar_queue_mining.model.job.job_to_vector_transformer import MinMaxScaler
from prolothar_queue_mining.model.job.job_to_vector_transformer import OneHotEncoder

class TestPairwisePriorityClassifierWaitingArea(unittest.TestCase):

    def test_add_and_pop_jobs(self):
        classifier = SklearnPairwisePriorityClassifier(
            LogisticRegression(), JobToVectorTransformer(
                ['size'], ['color'],
                {'size': MinMaxScaler(0, 100)},
                {'color': OneHotEncoder(['blue', 'yellow', 'red'])}
            ), scaler_for_arrival_time_difference=MinMaxScaler(-1, 1))

        waiting_area = PairwisePriorityClassifierWaitingArea(classifier)

        jobs = []
        arrival_times = []

        random_generator = Random(42)
        for i in range(100):
            arrival_times.append(random_generator.random())
            jobs.append(Job(str(i), {'size': i, 'color': random_generator.choice(['blue', 'yellow', 'red'])}))
            waiting_area.add_job_for_learning(arrival_times[-1], jobs[-1])
        for job in jobs:
            waiting_area.pop_next_job_for_learning(job.job_id)
        waiting_area.learn_classifier()

        for arrival_time, job in zip(arrival_times, jobs):
            waiting_area.add_job(arrival_time, job)
        for expected_job in jobs:
            self.assertEqual(expected_job, waiting_area.pop_next_job(len(waiting_area)))

if __name__ == '__main__':
    unittest.main()