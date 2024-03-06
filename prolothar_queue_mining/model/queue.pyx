from collections import Counter
from typing import List

from prolothar_queue_mining.model.waiting_area import FirstComeFirstServeWaitingArea
from prolothar_queue_mining.model.environment import Environment
from prolothar_queue_mining.model.event import Event
from prolothar_queue_mining.model.event cimport Event
from prolothar_queue_mining.model.arrival_process import ArrivalProcess
from prolothar_queue_mining.model.arrival_process import NullArrival
from prolothar_queue_mining.model.exit import Exit, DoNothingExit
from prolothar_queue_mining.model.waiting_area import WaitingArea
from prolothar_queue_mining.model.distribution import DiscreteDistribution
from prolothar_queue_mining.model.distribution import DiscreteDegenerateDistribution
from prolothar_queue_mining.model.observer.waiting_time import WaitingTimeObserver
from prolothar_queue_mining.model.observer.waiting_time import NullWaitingTimeObserver
from prolothar_queue_mining.model.observer.sojourn_time import SojournTimeObserver
from prolothar_queue_mining.model.observer.sojourn_time import NullSojournTimeObserver
from prolothar_queue_mining.model.observer.queue_length import QueueLengthObserver
from prolothar_queue_mining.model.observer.queue_length import NullQueueLengthObserver

cdef class Queue:

    def __init__(
            self, arrival_process: ArrivalProcess,
            servers: list[Server], exit_point: Exit = DoNothingExit(),
            waiting_area: WaitingArea = None,
            batch_size_distribution: DiscreteDistribution = DiscreteDegenerateDistribution(1),
            waiting_time_observer: WaitingTimeObserver = NullWaitingTimeObserver(),
            sojourn_time_observer: SojournTimeObserver = NullSojournTimeObserver(),
            queue_length_observer: QueueLengthObserver = NullQueueLengthObserver()):
        if arrival_process is None:
            arrival_process = NullArrival()
        self.__arrival_process = arrival_process
        self.__servers = servers
        if waiting_area is None:
            self.__waiting_area = FirstComeFirstServeWaitingArea()
        else:
            self.__waiting_area = waiting_area
        self.__exit = exit_point
        self.__waiting_time_observer = waiting_time_observer
        self.__arrival_time_of_open_jobs: dict[Job,float] = {}
        self.__sojourn_time_observer = sojourn_time_observer
        self.__queue_length_observer = queue_length_observer
        self.__batch_size_distribution = batch_size_distribution
        self.__current_required_batch_size = max(1, self.__batch_size_distribution.get_next_sample())
        self.__is_batch_service_possible = (
            self.__batch_size_distribution.get_mean() > 1 or
            self.__batch_size_distribution.get_variance() > 0)
        self.__nr_of_jobs_in_system = 0

    def set_seed(self, seed: int|None):
        """
        sets the seeds for distributions used by the parts in this queue
        """
        self.__arrival_process.set_seed(seed)
        self.__batch_size_distribution.set_seed(seed)
        for i,server in enumerate(self.__servers):
            if seed is not None:
                server.set_seed(seed + i)
            else:
                server.set_seed(seed)

    def set_waiting_time_observer(self, waiting_time_observer: WaitingTimeObserver):
        self.__waiting_time_observer = waiting_time_observer

    def get_waiting_time_observer(self) -> WaitingTimeObserver:
        return self.__waiting_time_observer

    def set_sojourn_time_observer(self, sojourn_time_observer: SojournTimeObserver):
        self.__sojourn_time_observer = sojourn_time_observer

    def get_sojourn_time_observer(self) -> SojournTimeObserver:
        return self.__sojourn_time_observer

    def set_queue_length_observer(self, queue_length_observer: QueueLengthObserver):
        self.__queue_length_observer = queue_length_observer

    def get_queue_length_observer(self) -> QueueLengthObserver:
        return self.__queue_length_observer

    def get_nr_of_servers(self) -> int:
        """
        returns the number of servers (c in Kendall's notation)
        """
        return len(self.__servers)

    def get_servers(self) -> list:
        """
        returns the list of servers in this queue
        """
        return self.__servers

    def add_server(self, server: Server):
        """
        adds a server to the end of the server list
        """
        self.__servers.append(server)

    def remove_server(self, int server_index):
        """
        removes the i-th server in this queue. the index can be negative,
        e.g. -1 to remove the last server.
        """
        del self.__servers[server_index]

    def get_service_time_name(self) -> str:
        """
        returns a short description of the service time distribution (S in Kendall's notation)
        """
        service_time_names = Counter(str(s.get_service_time_name()) for s in self.__servers)
        if len(service_time_names) == 1:
            return next(iter(service_time_names.keys()))
        else:
            return str(service_time_names)

    def get_batch_size_distribution(self) -> DiscreteDistribution:
        return self.__batch_size_distribution

    def get_arrival_process(self) -> ArrivalProcess:
        return self.__arrival_process

    def set_arrival_process(self, arrival: ArrivalProcess):
        if arrival is None:
            self.__arrival_process = NullArrival()
        else:
            self.__arrival_process = arrival

    def get_waiting_area(self) -> WaitingArea:
        return self.__waiting_area

    def set_waiting_area(self, waiting_area: WaitingArea) -> WaitingArea:
        self.__waiting_area = waiting_area

    def set_exit(self, exit_point: Exit):
        self.__exit = exit_point

    def get_exit(self) -> Exit:
        return self.__exit

    cpdef schedule_next_arrival(self, Environment environment):
        try:
            arrival_time, job = self.__arrival_process.get_next_job()
            environment.schedule_event(QueueArrivalEvent(self, arrival_time, job))
        except StopIteration:
            #there are no more elements in the population
            pass

    cpdef handle_job_arrival(self, Environment environment, int arrival_time, Job job):
        self.__arrival_time_of_open_jobs[job] = arrival_time
        self.__waiting_area.add_job(arrival_time, job)
        environment.schedule_event(QueueTryToServeEvent(self, environment.get_current_time()))
        self.__nr_of_jobs_in_system += 1
        self.schedule_next_arrival(environment)

    cpdef handle_job_exit(self, Environment environment, Server server, int exit_time, Job job):
        server.set_current_job(None)
        self.__exit.add_job(exit_time, job)
        self.__nr_of_jobs_in_system -= 1
        environment.schedule_event(QueueTryToServeEvent(self, environment.get_current_time()))
        self.__sojourn_time_observer.notify(job, self.__arrival_time_of_open_jobs.pop(job), exit_time)

    cdef handle_batch_exit(self, Environment environment, Server server, int exit_time, list batch: List[Job]):
        server.set_current_job(None)
        cdef Job job
        for job in batch:
            self.__exit.add_job(exit_time, job)
            self.__sojourn_time_observer.notify(job, self.__arrival_time_of_open_jobs.pop(job), exit_time)
        self.__nr_of_jobs_in_system -= len(batch)
        environment.schedule_event(QueueTryToServeEvent(self, environment.get_current_time()))

    cdef try_to_serve_next_job(self, Environment environment):
        if len(self.__waiting_area) < self.__current_required_batch_size:
            return
        cdef Server server
        cdef int exit_time
        cdef list batch
        cdef Job job
        for server in self.__servers:
            if server.is_ready_for_service():
                if self.__is_batch_service_possible:
                    batch = self.__waiting_area.pop_batch(
                        self.__current_required_batch_size, self.__nr_of_jobs_in_system)
                    exit_time = environment.get_current_time() + server.get_batch_service_time(
                        batch, self.__nr_of_jobs_in_system)
                    self.__serve_batch(batch, server, exit_time, environment)
                    self.__current_required_batch_size = max(1, self.__batch_size_distribution.get_next_sample())
                elif self.__waiting_area.has_next_job():
                    job = self.__waiting_area.pop_next_job(self.__nr_of_jobs_in_system)
                    exit_time = environment.get_current_time() + server.get_service_time(
                        job, self.__nr_of_jobs_in_system)
                    self.__serve_job(job, server, exit_time, environment)
                break
        self.__queue_length_observer.notify(environment.get_current_time(), len(self.__waiting_area))

    cdef __serve_job(self, Job job, Server server, int exit_time, Environment environment):
        self.__waiting_time_observer.notify(
            job, self.__arrival_time_of_open_jobs[job], environment.get_current_time())
        server.set_current_job(job)
        environment.schedule_event(QueueExitEvent(
            self, server, job, exit_time))

    cdef __serve_batch(self, list batch: List[Job], Server server, int exit_time, Environment environment):
        cdef Job job
        for job in batch:
            self.__waiting_time_observer.notify(
                job, self.__arrival_time_of_open_jobs[job], environment.get_current_time())
        #mark server as occupied
        server.set_current_job(job)
        environment.schedule_event(QueueBatchExitEvent(
            self, server, batch, exit_time))

    def copy(self) -> 'Queue':
        return Queue(
            self.__arrival_process.copy(),
            [s.copy() for s in self.__servers],
            exit_point=self.__exit.copy(),
            waiting_area=self.__waiting_area.copy(),
            batch_size_distribution=self.__batch_size_distribution.copy(),
            waiting_time_observer=self.__waiting_time_observer.copy(),
            sojourn_time_observer=self.__sojourn_time_observer.copy(),
            queue_length_observer=self.__queue_length_observer.copy()
        )

    def copy_mean(self) -> 'Queue':
        """
        copy the queue but removes all stochastic behavior, which is set to its
        mean
        """
        return Queue(
            self.__arrival_process.copy(),
            [s.copy_mean() for s in self.__servers],
            exit_point=self.__exit.copy(),
            waiting_area=self.__waiting_area.copy(),
            batch_size_distribution=DiscreteDegenerateDistribution(self.__batch_size_distribution.get_mean()),
            waiting_time_observer=self.__waiting_time_observer.copy(),
            sojourn_time_observer=self.__sojourn_time_observer.copy(),
            queue_length_observer=self.__queue_length_observer.copy()
        )

    def __repr__(self):
        return (
            'Queue(\n'
            f'    D={self.get_waiting_area().get_discipline_name()}\n'
            f'    c={self.get_nr_of_servers()}\n'
            f'    S={self.get_service_time_name()}\n'
            f'    BS={self.get_batch_size_distribution()}\n'
            ')'
        )

cdef class QueueArrivalEvent(Event):
    cdef Queue __queue
    cdef Job __job

    def __init__(self, Queue queue, int arrival_time, Job job):
        super().__init__(arrival_time)
        self.__queue = queue
        self.__job = job

    cpdef execute(self, Environment environment):
        self.__queue.handle_job_arrival(environment, self.time, self.__job)

    def __repr__(self):
        return f'{self.time} - QueueArrival of {self.__job.job_id}'

cdef class QueueTryToServeEvent(Event):
    cdef Queue __queue

    def __init__(self, Queue queue, int time):
        super().__init__(time, prio=1)
        self.__queue = queue

    cpdef execute(self, Environment environment):
        self.__queue.try_to_serve_next_job(environment)

    def __repr__(self):
        return f'{self.time} - Try to serve'

cdef class QueueExitEvent(Event):
    cdef Queue __queue
    cdef Job __job
    cdef Server __server

    def __init__(self, Queue queue, Server server, Job job, int exit_time):
        super().__init__(exit_time)
        self.__queue = queue
        self.__job = job
        self.__server = server

    cpdef execute(self, Environment environment):
        self.__queue.handle_job_exit(environment, self.__server, self.time, self.__job)

    def __repr__(self):
        return f'{self.time} - QueueExit of {self.__job.job_id}'

cdef class QueueBatchExitEvent(Event):

    cdef Queue __queue
    cdef Server __server
    cdef list __batch

    def __init__(self, Queue queue, Server server, list batch: List[Job], int exit_time):
        super().__init__(exit_time)
        self.__queue = queue
        self.__batch = batch
        self.__server = server

    cpdef execute(self, Environment environment):
        self.__queue.handle_batch_exit(environment, self.__server, self.time, self.__batch)

    def __repr__(self):
        return f'{self.time} - QueueExit of {[job.job_id for job in self.__batch]}'
