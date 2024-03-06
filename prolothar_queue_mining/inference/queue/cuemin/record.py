from dataclasses import dataclass

from prolothar_queue_mining.model.service_time import ServiceTime

@dataclass
class Record:
    """
    recording of quality score dependent on the input parameter space
    """
    waiting_area: str
    batch_size_distribution: str
    nr_of_servers: int
    service_time: ServiceTime
    mdl_model: float
    mdl_service_time: float
    mdl_service_time_values: float
    mdl_service_time_residual: float
    mdl_batching: float
    mdl_score: float