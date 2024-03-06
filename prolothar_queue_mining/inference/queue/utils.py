from typing import Iterable

from statistics import quantiles

from prolothar_queue_mining.model.distribution import ContinuousDistribution
from prolothar_queue_mining.model.distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.distribution import GeometricDistribution
from prolothar_queue_mining.model.distribution import NegativeBinomialDistribution
from prolothar_queue_mining.model.distribution import PoissonDistribution
from prolothar_queue_mining.model.distribution import C2dDistribution
from prolothar_queue_mining.model.job import Job

"""
bundles utility functions for queue inference
"""

ZERO_ONLY_DISTRIBUTION = DiscreteDegenerateDistribution(0)

def generate_distribution_candidates(
    observations: list[int]|list[float],
    seed_for_distributions: float=None) -> Iterable[DiscreteDistribution]:
    """
    parameter inference for a pre-defined set of discrete distributions on the
    given list of values

    Parameters
    ----------
    observations : list[float]
        list of observed values on which the distributions are supposed to be fitted
    seed : float, optional
        seeds for the distributions, by default None

    Yields
    -------
    Generator[DiscreteDistribution, None, None]
        yields MLE fitted distributions
    """
    distribution_fitter_list = []
    # distribution_fitter_list.append(lambda d: d.fit(observations, seed=seed_for_distributions))
    if len(observations) > 2:
        quantile_list = quantiles(observations, n=20)
        filtered_observations = []
        for value in sorted(observations):
            if quantile_list[0] <= value <= quantile_list[-1]:
                filtered_observations.append(value)
        distribution_fitter_list.append(lambda d: d.fit(filtered_observations, seed=seed_for_distributions))
    if any(o <= 0 for o in observations):
        positive_observations = [o for o in observations if o > 0]
        distribution_fitter_list.append(lambda d: d.fit(positive_observations, seed=seed_for_distributions))

    distribution_candidates = set()
    for distribution in [
            DiscreteDegenerateDistribution, GeometricDistribution,
            NegativeBinomialDistribution, PoissonDistribution]:
        for distribution_fitter in distribution_fitter_list:
            try:
                fitted_distribution = distribution_fitter(distribution)
                if isinstance(fitted_distribution, ContinuousDistribution):
                    fitted_distribution = C2dDistribution(fitted_distribution)
                if is_valid_distribution_candidate(fitted_distribution):
                    distribution_candidates.add(fitted_distribution)
            except (ValueError, FloatingPointError, ZeroDivisionError, RuntimeError):
                #skip distribution that cannot be fitted
                pass
    return distribution_candidates

def is_valid_distribution_candidate(fitted_distribution: DiscreteDistribution) -> bool:
    #make sure that a non-degenerate distribution is not degenerate
    if not isinstance(fitted_distribution, DiscreteDegenerateDistribution) and fitted_distribution.get_variance() == 0:
        return False
    if fitted_distribution == ZERO_ONLY_DISTRIBUTION:
        return False
    if fitted_distribution.get_mode() < 0 or fitted_distribution.get_mean() < 0:
        return False
    #filter out distributions with extreme parameter values
    try:
        fitted_distribution.get_mdl_of_model()
    except OverflowError:
        return False
    return True

def count_nr_of_jobs_in_system(
        arrivals: dict[Job, int], departures: dict[Job, int]) -> tuple[list[int], list[int]]:
    arrival_list = sorted(arrivals.values(), reverse=True)
    departure_list = sorted(departures.values(), reverse=True)
    current_arrival_time = round(arrival_list.pop()) if arrivals else float('inf')
    current_departure_time = round(departure_list.pop()) if departures else float('inf')
    current_nr_of_jobs = 0
    last_nr_of_jobs = 0
    last_t = 0
    x = []
    y = []
    while current_arrival_time < float('inf') or current_departure_time < float('inf'):
        if current_arrival_time < current_departure_time:
            current_nr_of_jobs += 1
            current_t = current_arrival_time
            try:
                current_arrival_time = round(arrival_list.pop())
            except IndexError:
                current_arrival_time = float('inf')
        else:
            current_nr_of_jobs -= 1
            current_t = current_departure_time
            try:
                current_departure_time = round(departure_list.pop())
            except IndexError:
                current_departure_time = float('inf')
        if current_t != last_t and current_t != float('inf'):
            for t in range(last_t, current_t):
                x.append(t)
                y.append(last_nr_of_jobs)
            last_nr_of_jobs = current_nr_of_jobs
            last_t = current_t
    return x,y
