from prolothar_queue_mining.model.job import Job

def create_job_departure_order_list(
    observed_arrivals: list[tuple[Job, float]],
    observed_departures: list[tuple[Job, float]]) -> list[int]:
    """
    creates a list of arrival indices ordered by departure time.

    Parameters
    ----------
    observed_arrivals : list[tuple[Job, float]]
        a list of jobs and corresponding arrival times
    observed_departures : list[tuple[Job, float]]
        a list of jobs and corresponding departure times

    Returns
    -------
    list[int]
        a list of arrival indices ordered by departure time.
        the initial job list is [1,2,3,4,...,nr_of_jobs].
        this list is sorted by the departure time of each job.
    """
    job_departure_dict = {}
    for job, departure_time in observed_departures:
        job_departure_dict[job] = departure_time
    job_arrival_departure_list = []
    i = 0
    last_arrival_time = None
    for job, arrival_time in observed_arrivals:
        if job in job_departure_dict:
            if arrival_time != last_arrival_time:
                i += 1
                last_arrival_time = arrival_time
            job_arrival_departure_list.append((i, arrival_time, job_departure_dict[job]))
    job_arrival_departure_list.sort(key=lambda x: x[2])
    return [i for i,_,_ in job_arrival_departure_list]

def create_combined_job_index_list(
    observed_arrivals: list[tuple[Job, float]],
    observed_departures: list[tuple[Job, float]]) -> list[int]:
    """
    creates a list of job indices ordered by event time. the resulting list contains each
    job index twice (arrival + departure). job indices increase with arrival time.

    Parameters
    ----------
    observed_arrivals : list[tuple[Job, float]]
        a list of jobs and corresponding arrival times
    observed_departures : list[tuple[Job, float]]
        a list of jobs and corresponding departure times

    Returns
    -------
    list[int]
        a list of job indices ordered by event time. the resulting list contains each
        job index twice (arrival + departure).
    """
    job_indices_dict = {x[0]: i+1 for i,x in enumerate(observed_arrivals)}
    combined_list = sorted(observed_arrivals + observed_departures, key=lambda x: x[1])
    return [job_indices_dict[job] for job,_ in combined_list]
