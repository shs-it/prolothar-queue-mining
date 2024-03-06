from typing import Tuple, List
from itertools import chain

from prolothar_queue_mining.model.job import Job
from prolothar_queue_mining.model.queue cimport Queue
from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.population import ListPopulation
from prolothar_queue_mining.model.server cimport Server
from prolothar_queue_mining.model.server cimport ListRecordingServer
from prolothar_queue_mining.model.service_time import OracleServiceTime
from prolothar_queue_mining.model.environment cimport Environment
from prolothar_queue_mining.model.distribution import PseudoDistribution
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.observer.waiting_time import WaitingTimeRecordingObserver
from prolothar_queue_mining.model.observer.sojourn_time import SojournTimeRecordingObserver
from prolothar_queue_mining.model.exit import ListCollectorExit
from prolothar_queue_mining.inference.queue.batch import LargestGapBatchMiner

cpdef tuple infer_waiting_and_service_times(
    list observed_arrivals: List[Tuple[Job, int]],
    list observed_departures: List[Tuple[Job, int]],
    waiting_area: WaitingArea,
    int nr_of_servers,
    bint filter_delayed_jobs = False):
    """
    infers optimal waiting and service times given observed arrivals and
    departures and a number of available servers. this method assumes that
    there is no downtime of servers.

    returns Tuple[Dict[Job, int], Dict[Job, int], List[List[Job]]]
    - 0: assigns the waiting time to each job
    - 1: assigns a service time to each job. if a job does not leave the queue, there
    will be no service time for it. if arrival of a job is unknown, there will be
    neither a waiting time nor a service time.
    - 2: gives the serving order for each server in the queue

    """
    cdef dict exit_time_per_job = {job: exit_time for job, exit_time in observed_departures}
    cdef dict arrival_time_per_job = {job: arrival_time for job, arrival_time in observed_arrivals}
    cdef dict sojourn_time_per_job = {
        job: exit_time_per_job[job] - arrival_time for job, arrival_time in observed_arrivals
        if job in exit_time_per_job
    }
    cdef Environment environment = Environment()
    cdef Queue queue = Queue(
        FixedArrival(
            ListPopulation([job for job,_ in observed_arrivals if job in exit_time_per_job]),
            [arrival_time for job,arrival_time in observed_arrivals if job in exit_time_per_job]
        ),
        [ListRecordingServer(OracleServiceTime(environment, exit_time_per_job)) for _ in range(nr_of_servers)],
        waiting_area=waiting_area.copy(),
        waiting_time_observer=WaitingTimeRecordingObserver()
    )
    queue.schedule_next_arrival(environment)
    environment.run_timesteps(observed_departures[-1][1])
    cdef dict waiting_time_per_job = queue.get_waiting_time_observer().get_waiting_time_per_job_dict()
    service_time_per_job = {
        job: sojourn_time_per_job[job] - waiting_time
        for job, waiting_time in waiting_time_per_job.items()
        if not filter_delayed_jobs or waiting_time <= sojourn_time_per_job[job]
    }
    return waiting_time_per_job, service_time_per_job, [s.get_served_jobs() for s in queue.get_servers()]

cpdef tuple infer_service_times_batch(
        list observed_arrivals: List[Tuple[Job, int]],
        list observed_departures: List[Tuple[Job, int]],
        waiting_area: WaitingArea,
        int nr_of_servers):
    """
    infers batch service times given observed arrivals and
    departures and a number of available servers. this method assumes that
    there is no downtime of servers. furthermore, this method assumes that jobs
    with the same departure time have been processed in one batch.

    returns three lists:
    1) contains jobs grouped in batches. jobs are in the same batch if they have the same original departure time,
    2) contains jobs grouped in batches. jobs are in the same batch if they have the same reconstructed departure time
    3) contains the corresponding computed service times

    returns tuple[list[list[Job]], tuple[list[list[Job]], list[int]]
    """
    cdef dict exit_time_per_job = {job: exit_time for job, exit_time in observed_departures}
    # cdef list observed_batches = LargestGapBatchMiner().group_batches(observed_arrivals, observed_departures)
    cdef list observed_batches = infer_batches(observed_departures)
    cdef Environment environment = Environment()
    cdef Queue queue = Queue(
        FixedArrival(
            ListPopulation([job for job,_ in observed_arrivals if job in exit_time_per_job]),
            [arrival_time for job,arrival_time in observed_arrivals if job in exit_time_per_job]
        ),
        [Server(OracleServiceTime(environment, exit_time_per_job)) for _ in range(nr_of_servers)],
        waiting_area=waiting_area.copy(),
        batch_size_distribution=PseudoDistribution([len(b) for b in observed_batches]),
        waiting_time_observer=WaitingTimeRecordingObserver(),
        sojourn_time_observer=SojournTimeRecordingObserver(),
        exit_point=ListCollectorExit()
    )
    queue.schedule_next_arrival(environment)
    environment.run_until_event_queue_is_empty()

    cdef dict waiting_time_per_job = queue.get_waiting_time_observer().get_waiting_time_per_job_dict()
    cdef dict sojourn_time_per_job = queue.get_sojourn_time_observer().get_sojourn_time_per_job_dict()

    cdef list service_times = []
    cdef list batches = infer_batches([
        (job, exit_time) for job, exit_time in zip(*queue.get_exit().get_recording())
    ])
    cdef list batch
    for batch in batches:
        service_times.append(sojourn_time_per_job[batch[-1]] - waiting_time_per_job[batch[-1]])

    return observed_batches, batches, service_times

cdef list infer_batches(list observed_departures: List[Tuple[Job, int]]):
    cdef list batches = []
    cdef list current_batch = [observed_departures[0][0]]
    cdef int current_departure_time = observed_departures[0][1]
    for job, departure_time in chain(observed_departures[1:], [(None, -1)]):
        if departure_time != current_departure_time:
            batches.append(current_batch)
            current_departure_time = departure_time
            current_batch = []
        current_batch.append(job)
    return batches
