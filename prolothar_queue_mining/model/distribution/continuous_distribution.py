from abc import abstractmethod

from prolothar_queue_mining.model.distribution.distribution import Distribution

class ContinuousDistribution(Distribution):
    """
    template of a continuous distribution, from which we can sample random numbers
    """

    @abstractmethod
    def compute_pdf(self, x: float) -> float:
        """
        computes the value of the PDF at x
        """
