from typing import Tuple, Dict, List

cpdef tuple infer_waiting_and_service_times(
    list observed_arrivals: List[Tuple[Job, float]],
    list observed_departures: List[Tuple[Job, float]],
    waiting_area: WaitingArea,
    int nr_of_servers,
    bint filter_delayed_jobs = *)

cpdef tuple infer_service_times_batch(
    list observed_arrivals: List[Tuple[Job, float]],
    list observed_departures: List[Tuple[Job, float]],
    waiting_area: WaitingArea,
    int nr_of_servers)