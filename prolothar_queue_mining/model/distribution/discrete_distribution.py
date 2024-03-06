from abc import abstractmethod

from prolothar_queue_mining.model.distribution.distribution import Distribution

class DiscreteDistribution(Distribution):
    """
    template of a distribution, from which we can sample random numbers
    """

    @abstractmethod
    def compute_pmf(self, x: float) -> float:
        """
        computes the value of the probability mass function at x.
        """

    @abstractmethod
    def get_mdl_of_model(self) -> float:
        """
        computes and returns the model complexity of this distribution
        """

    @abstractmethod
    def is_deterministic(self) -> bool:
        """
        returns True iff there is one value with PMF=1, i.e. sampling from
        the distribution will always return the same value
        """

    @classmethod
    def fit_remove_outliers(cls, data: list[float], seed: int|None = None) -> 'DiscreteDistribution':
        """
        fits parameters with the following scheme:
        fit() is used
        then all observations with pmf <= ALLMOST_ZERO are removed
        fit() is used again on remaining observations
        """
        distribution = cls.fit(data, seed=seed)
        filtered_observations = [
            x for x in data
            if distribution.compute_pmf(x) > Distribution.ALLMOST_ZERO
        ]
        if filtered_observations:
            return cls.fit(
                filtered_observations,
                seed=seed
            )
        else:
            return distribution
