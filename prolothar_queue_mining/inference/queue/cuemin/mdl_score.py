from math import log2
from prolothar_common.mdl_utils import L_N
from prolothar_queue_mining.model.distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.job import Job

from prolothar_queue_mining.model.arrival_process import FixedArrival
from prolothar_queue_mining.model.service_time import ServiceTime
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.server import Server

from prolothar_queue_mining.inference.queue.cuemin.mdl_service_time import MdlServiceTime
from prolothar_queue_mining.inference.queue.cuemin.mdl_batch_size_distribution import MdlBatchSizeDistribution
from prolothar_queue_mining.inference.queue.times import infer_waiting_and_service_times
from prolothar_queue_mining.inference.queue.times import infer_service_times_batch

BATCHSIZE_ONE_DISTRIBUTION = DiscreteDegenerateDistribution(1)

def compute_length_of_model(
        waiting_area: WaitingArea, nr_of_servers: int,
        service_time: ServiceTime, batch_size_distribution: DiscreteDistribution,
        nr_of_categorical_attributes: int) -> float:
    """
    computes L(M)
    """
    mdl_model = waiting_area.get_mdl(nr_of_categorical_attributes)
    mdl_model += L_N(nr_of_servers)
    mdl_model += service_time.get_mdl_of_model()
    if batch_size_distribution != BATCHSIZE_ONE_DISTRIBUTION:
        mdl_model += batch_size_distribution.get_mdl_of_model()
    return mdl_model

def compute_lower_bound_implied_by_model(
        waiting_area: WaitingArea, nr_of_servers: int,
        service_time: ServiceTime, batch_size_distribution: DiscreteDistribution,
        nr_of_jobs: int, nr_of_categorical_attributes: int):
    """
    fast-to-compute lower bound for L(M) + L(D|M). it is computed by assuming
    that all "nr_of_jobs" jobs are encoded with the shortest code implied by the model.
    """
    return (
        compute_length_of_model(
            waiting_area, nr_of_servers, service_time,
            batch_size_distribution, nr_of_categorical_attributes) +
        int(nr_of_jobs / batch_size_distribution.get_mean()) *
        service_time.get_min_code_length_for_one_job()
    )

def compute_lower_bound_implied_by_model_and_data(
        waiting_area: WaitingArea, nr_of_servers: int,
        service_time_model: ServiceTime, batch_size_distribution: DiscreteDistribution,
        no_batching_service_time_histogram: dict[int, int],
        batching_service_time_histogram: dict[int, int],
        batch_size_histogram: dict[int, int],
        nr_of_categorical_attributes: int):
    """
    computes a lower bound for L(M) + L(D|M). this is slower than
    "compute_lower_bound_implied_by_model" but also tighter
    """
    lower_bound = compute_length_of_model(
        waiting_area, nr_of_servers, service_time_model,
        batch_size_distribution, nr_of_categorical_attributes)
    if batch_size_distribution.is_deterministic() and batch_size_distribution.get_mean() == 1:
        for service_time, frequency in no_batching_service_time_histogram.items():
            max_probability = service_time_model.compute_max_probability(service_time)
            if max_probability > Distribution.ALLMOST_ZERO:
                lower_bound -= frequency * log2(max_probability)
            else:
                lower_bound += frequency * L_N(abs(service_time)+1)
    else:
        for service_time, frequency in batching_service_time_histogram.items():
            max_probability = service_time_model.compute_max_probability(service_time)
            if max_probability > Distribution.ALLMOST_ZERO:
                lower_bound -= frequency * log2(max_probability)
            else:
                lower_bound += frequency * L_N(abs(service_time)+1)
        if not batch_size_distribution.is_deterministic():
            for batch_size, frequency in batch_size_histogram.items():
                probability = batch_size_distribution.compute_pmf(batch_size)
                if probability > Distribution.ALLMOST_ZERO:
                    lower_bound -= frequency * log2(probability)
                else:
                    lower_bound -= frequency * log2(batch_size_distribution.compute_pmf(
                        batch_size_distribution.get_mode()))
    return lower_bound

def compute_mdl(
        queue: Queue,
        arrivals: list[tuple[Job, int]],
        departures: list[tuple[Job, int]],
        departure_time_per_job: dict[Job, int],
        nr_of_categorical_attributes: int) -> float:
    environment = Environment(verbose=False)
    waiting_area = queue.get_waiting_area().copy_empty()
    raw_service_time_model = queue.get_servers()[0].get_service_time_definition().copy()
    mdl_service_time = MdlServiceTime(
        environment,
        raw_service_time_model,
        departure_time_per_job)
    batches, _, _ = infer_service_times_batch(
        arrivals, departures, waiting_area,
        queue.get_nr_of_servers())
    observed_batch_sizes = [len(b) for b in batches]
    batch_size_distribution = MdlBatchSizeDistribution(
        queue.get_batch_size_distribution().copy(),
        observed_batch_sizes)
    queue = Queue(
        FixedArrival.create_from_observation(arrivals),
        [Server(mdl_service_time) for _ in range(queue.get_nr_of_servers())],
        waiting_area=waiting_area,
        batch_size_distribution=batch_size_distribution)
    queue.schedule_next_arrival(environment)
    environment.run_until_event_queue_is_empty()
    while queue.get_waiting_area().has_next_job():
        mdl_service_time.get_service_time(
            queue.get_waiting_area().pop_next_job(len(queue.get_waiting_area())),
            len(queue.get_waiting_area()))

    candidate_mdl_score_model = compute_length_of_model(
        queue.get_waiting_area(), queue.get_nr_of_servers(), raw_service_time_model,
        batch_size_distribution.get_distribution(), nr_of_categorical_attributes)
    candidate_mdl_score_service_time = mdl_service_time.get_total_encoded_length()
    candidate_mdl_score_batch_distribution = batch_size_distribution.get_total_encoded_length()
    return (
        candidate_mdl_score_model +
        candidate_mdl_score_service_time +
        candidate_mdl_score_batch_distribution
    )
