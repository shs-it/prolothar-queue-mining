from prolothar_queue_mining.model.arrival_process import ArrivalProcess
from prolothar_queue_mining.model.arrival_process import NullArrival
from prolothar_queue_mining.model.exit import Exit
from prolothar_queue_mining.model.server import Server
from prolothar_queue_mining.model.queue import Queue
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.event import Event
from prolothar_queue_mining.model.queue_network.router import Router
from prolothar_queue_mining.model.job import Job

class QueueNetwork:

    def __init__(self, environment: Environment, arrival: ArrivalProcess, exit_point: Exit, router: Router):
        self.__arrival = arrival
        self.__exit_point = exit_point
        self.__queue_dict = {}
        self.__node_exit_pipe = NodeExitPipe(self, environment)
        self.__router = router

    def add_queue_node(self, queue_name: str, servers: list[Server], waiting_area: WaitingArea = None):
        self.__queue_dict[queue_name] = Queue(
            NullArrival(), servers, exit_point=self.__node_exit_pipe,
            waiting_area=waiting_area
        )

    def schedule_next_arrival(self, environment: Environment):
        try:
            arrival_time, job = self.__arrival.get_next_job()
            environment.schedule_event(ArrivalEvent(self, arrival_time, job))
        except StopIteration:
            #there are no more elements in the population
            pass

    def handle_job_arrival(self, environment: Environment, arrival_time: int, job: Job):
        self.schedule_queue_transition(environment, arrival_time, job)
        self.schedule_next_arrival(environment)

    def schedule_queue_transition(self, environment: Environment, timestep: int, job: Job):
        try:
            queue = self.__queue_dict[self.__router.get_name_of_next_queue(job, timestep)]
            queue.handle_job_arrival(environment, timestep, job)
        except StopIteration:
            self.__exit_point.add_job(timestep, job)

    def get_queue_dict(self) -> dict[str, Queue]:
        return self.__queue_dict

class ArrivalEvent(Event):

    def __init__(self, queue_network: QueueNetwork, arrival_time: int, job: Job):
        super().__init__(arrival_time)
        self.__queue_network = queue_network
        self.__job = job

    def execute(self, environment: Environment):
        self.__queue_network.handle_job_arrival(environment, self.time, self.__job)

    def __repr__(self):
        return f'{self.time} - Arrival of {self.__job.job_id}'

class NodeExitPipe(Exit):

    def __init__(self, queue_network: QueueNetwork, environment: Environment):
        self.__queue_network = queue_network
        self.__environment = environment

    def add_job(self, timestep: int, job: Job):
        self.__queue_network.schedule_queue_transition(self.__environment, timestep, job)
