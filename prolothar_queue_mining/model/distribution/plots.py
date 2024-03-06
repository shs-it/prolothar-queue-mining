from typing import Iterable

from prolothar_common.experiments.plots.plot_context import PlotContext

from prolothar_queue_mining.model.distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution import ContinuousDistribution

def plot_pmf(
    distribution: DiscreteDistribution, x_values: Iterable[float],
    show: bool = True, filepath: str = None):
    """
    plots the PMF of the given distribution

    Parameters
    ----------
    distribution : DiscreteDistribution
        distribution for which the PMF is supposed to be plotted
    x_values : Iterable[float]
        the points at which the PMF should be evaluated and plotted
    show : bool, optional
        whether the plot should be shown, by default True
    filepath : str, optional
        can be supplied to save the plot into a file, by default None
    """
    with PlotContext(show=show, filepath=filepath) as plot_context:
        plot_context.get_axes().bar(
            x_values, [distribution.compute_pmf(x) for x in x_values])

def plot_pmfs(
    distribution_list: list[DiscreteDistribution], x_values: Iterable[float],
    show: bool = True, filepath: str = None, legend_labels: list[str] = None,
    mode: str = 'scatter'):
    """
    plots the PMF of a list of distributions

    Parameters
    ----------
    distribution_list : list[DiscreteDistribution]
        distribution for which the PMF is supposed to be plotted
    x_values : Iterable[float]
        the points at which the PMF should be evaluated and plotted
    show : bool, optional
        whether the plot should be shown, by default True
    filepath : str, optional
        can be supplied to save the plot into a file, by default None
    alpha : float, optional
        value between 0 and 1. controls transparency of the scatter plot markers.
        1 means no transparency. default is 0.3
    """
    if legend_labels is None:
        legend_labels = [str(distribution) for distribution in distribution_list]
    with PlotContext(show=show, filepath=filepath) as plot_context:
        for distribution in distribution_list:
            y_values = [distribution.compute_pmf(x) for x in x_values]
            if mode == 'scatter':
                plot_context.get_axes().scatter(x_values, y_values)
            elif mode == 'line':
                plot_context.get_axes().plot(x_values, y_values)
            else:
                raise ValueError(f'unsupported mode {mode}')
        plot_context.get_axes().legend(legend_labels)


def plot_pdf(
    distribution: ContinuousDistribution, x_values: Iterable[float],
    show: bool = True, filepath: str = None):
    """
    plots the PdF of the given distribution

    Parameters
    ----------
    distribution : DiscreteDistribution
        distribution for which the PDF is supposed to be plotted
    x_values : Iterable[float]
        the points at which the PDF should be evaluated
    show : bool, optional
        whether the plot should be shown, by default True
    filepath : str, optional
        can be supplied to save the plot into a file, by default None
    """
    with PlotContext(show=show, filepath=filepath) as plot_context:
        plot_context.get_axes().plot(
            x_values, [distribution.compute_pdf(x) for x in x_values])