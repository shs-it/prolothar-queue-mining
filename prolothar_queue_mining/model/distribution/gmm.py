from math import sqrt
import numpy as np
from sklearn.mixture import GaussianMixture

from prolothar_common import mdl_utils

from prolothar_queue_mining.model.distribution.distribution import Distribution
from prolothar_queue_mining.model.distribution.continuous_distribution import ContinuousDistribution
from prolothar_queue_mining.model.distribution.normal import NormalDistribution
from prolothar_queue_mining.model.distribution.pmf_defined import PmfDefinedDistribution

class GaussianMixtureModelDistribution(ContinuousDistribution):
    """
    a mixture of normal or Gaussian distribution
    """

    def __init__(self, weights: list[float], means: list[float], variances: list[float], seed: int|None = None):
        """
        creates a new normal distribution with the given mean and standard deviation

        Parameters
        ----------
        mean : float
        stddev : float
        seed : float, optional
            random generator seed, by default None
        """
        self.__weights = weights
        self.__means = means
        self.__variances = variances

        self.__gaussians = [
            NormalDistribution(mean, sqrt(variance), seed=seed)
            for mean, variance in zip(means, variances)]
        self.__mean = 0
        for weight, gaussian in zip(self.__weights, self.__gaussians):
            self.__mean += weight * gaussian.get_mean()
        self.__gaussian_select_distribution = PmfDefinedDistribution(
            { i: w for i,w in enumerate(weights) }, seed=seed)

        mode_density = 0
        for weight, gaussian in zip(self.__weights, self.__gaussians):
            candidate_density = weight * gaussian.compute_pdf(gaussian.get_mode())
            if candidate_density > mode_density:
                mode_density = candidate_density
                self.__mode = gaussian.get_mode()
        self.seed = seed

    def get_next_sample(self) -> float:
        return self.__gaussians[self.__gaussian_select_distribution.get_next_sample()].get_next_sample()

    def get_mean(self) -> float:
        return self.__mean

    def get_mode(self) -> float:
        return self.__mode

    def get_variance(self) -> float:
        variance = -self.__mean**2
        for weight, gaussian in zip(self.__weights, self.__gaussians):
            variance += weight * (gaussian.get_variance() + gaussian.get_mean()**2)
        return variance

    def compute_pdf(self, x: float) -> float:
        pdf = 0
        for weight, gaussian in zip(self.__weights, self.__gaussians):
            pdf += weight * gaussian.compute_pdf(x)
        return pdf

    def get_mdl_of_model(self) -> float:
        mdl_of_model = mdl_utils.L_N(len(self.__gaussians))
        for gaussian in self.__gaussians:
            mdl_of_model += gaussian.get_mdl_of_model()
        return mdl_of_model

    def compute_cdf(self, x: float) -> float:
        cdf = 0
        for weight, gaussian in zip(self.__weights, self.__gaussians):
            try:
                cdf += weight * gaussian.compute_cdf(x)
            except FloatingPointError:
                cdf += ContinuousDistribution.ALLMOST_ZERO
        return cdf

    def copy(self) -> ContinuousDistribution:
        return GaussianMixtureModelDistribution(self.__weights, self.__means, self.__variances, seed=self.seed)

    def __repr__(self):
        return f'GmmDistribution({self.__weights}, {self.__means}, {self.__variances}, seed={self.seed})'

    def set_seed(self, seed: int):
        self.seed = seed
        self.__gaussian_select_distribution.set_seed(seed)
        for gaussian in self.__gaussians:
            gaussian.set_seed(seed)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return (
            isinstance(other, GaussianMixtureModelDistribution) and
            self.__weights == other.__weights and
            self.__means == other.__means and
            self.__variances == other.__variances
        )

    @staticmethod
    def fit(data: list[float], seed: int|None = None) -> 'Distribution':
        last_bic = float('inf')
        X = np.array(data).reshape(-1, 1)
        n_components = 1
        while(True):
            gmm = GaussianMixture(n_components=n_components, random_state=seed)
            gmm.fit(X)
            current_bic = gmm.bic(X)
            if current_bic > last_bic:
                return GaussianMixtureModelDistribution(
                    best_weights.flatten().tolist(),
                    best_means.flatten().tolist(),
                    best_variances.flatten().tolist()
                )
            best_weights = gmm.weights_
            best_means = gmm.means_
            best_variances = gmm.covariances_
            n_components += 1
            last_bic = current_bic

    @staticmethod
    def fit_by_mean_and_variance(mean: float, variance: float, seed: int|None = None) -> 'Distribution':
        raise NotImplementedError()