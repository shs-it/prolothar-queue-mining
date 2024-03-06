from collections import deque
from prolothar_queue_mining.model.queue_network.router.router import Router
from prolothar_queue_mining.model.job import Job

class StaticRouter(Router):
    """
    a router where a job follows a static list of stations
    """

    def __init__(self, routing_table: dict[object, list[str]]):
        """
        initializes this static router

        Parameters
        ----------
        routing_table : dict[object, list[str]]
            table that assigns a list of queue names to each job
        """
        self.__current_routing_table = {
            job: deque(route) for job,route in routing_table.items()
        }

    def get_name_of_next_queue(self, job: Job, current_time: int) -> str:
        try:
            return self.__current_routing_table[job].popleft()
        except IndexError as e:
            raise StopIteration() from e
