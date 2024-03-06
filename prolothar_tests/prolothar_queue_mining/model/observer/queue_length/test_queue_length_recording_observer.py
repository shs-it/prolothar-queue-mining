import unittest

from prolothar_queue_mining.model.observer.queue_length import QueueLengthRecordingObserver

class TestQueueLengthRecordingObserver(unittest.TestCase):

    def test_add_and_pop_jobs(self):
        observer = QueueLengthRecordingObserver()
        observer.notify(10, 0)
        observer.notify(15, 1)
        observer.notify(17, 2)
        observer.notify(19, 3)
        observer.notify(21, 2)
        observer.notify(25, 4)
        observer.notify(25, 6)

        self.assertEqual(6, observer.get_max_queue_length())
        timesteps,queue_lengths = observer.get_timeseries_data()
        self.assertEqual([10,15,17,19,21,25], timesteps)
        self.assertEqual([0,1,2,3,2,6], queue_lengths)

if __name__ == '__main__':
    unittest.main()