from abc import ABC, abstractmethod

from math import nextafter

class Distribution(ABC):
    """
    template of a distribution, from which we can sample random numbers
    """
    ALLMOST_ZERO = nextafter(0, 1)

    @abstractmethod
    def get_next_sample(self) -> float:
        """
        draws the next sample from this distribution
        """

    @abstractmethod
    def copy(self) -> 'Distribution':
        """
        creates a copy of this distribution. the internal state is not copied,
        i.e. the copy needs not to show necessarily the same behavior
        """

    @abstractmethod
    def __repr__(self) -> str:
        """
        returns a readable representation of this distribution
        """

    @abstractmethod
    def __hash__(self):
        """
        computes the hashcodes from the parameters of this distribution
        """

    @abstractmethod
    def __eq__(self, other):
        """
        two distributions are considered equal iff they are from the same type and
        share the same parameter values. this means, even if two distributions
        from different types behave exactly the same, they are considered different.
        """

    @abstractmethod
    def get_mean(self) -> float:
        """
        returns the mean of the distribution
        """

    @abstractmethod
    def get_mode(self) -> float:
        """
        returns the mode of the distribution, i.e. the most likely sample.
        some distributions do not have a unique mode. in such cases there is
        no guarantee on the returned value
        """

    @abstractmethod
    def get_variance(self) -> float:
        """
        returns the variance of the distribution
        """

    @abstractmethod
    def compute_cdf(self, x: float) -> float:
        """
        computes the value of the CDF at x
        """

    @abstractmethod
    def set_seed(self, seed: int):
        """
        sets / reinitializes the seed of any random generator used in this
        distribution
        """

    @staticmethod
    @abstractmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        """
        estimates the parameter of the distribution from given data
        """

    @staticmethod
    @abstractmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        """
        estimates the parameter of the distribution from mean and variance.
        raises a NotImplementedError if type of distribution does not support this fit
        """
