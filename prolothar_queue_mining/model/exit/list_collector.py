from prolothar_queue_mining.model.exit.exit import Exit
from prolothar_queue_mining.model.job import Job

class ListCollectorExit(Exit):
    """
    exit point that collects timesteps and jobs in a list
    """

    def __init__(self):
        self.__timesteps = []
        self.__jobs = []

    def add_job(self, timestep: int, job: Job):
        self.__timesteps.append(timestep)
        self.__jobs.append(job)

    def get_recording(self) -> tuple[list[Job], list[int]]:
        """
        returns two lists. the first one is the list of all jobs that have exited
        the queue and the second one is the list of the corresponding exit time
        stamps
        """
        return self.__jobs, self.__timesteps

    def __getitem__(self, index: int) -> tuple[int,Job]:
        return self.__timesteps[index], self.__jobs[index]

    def __str__(self) -> str:
        s = ''
        for timestep, job in zip(self.__timesteps, self.__jobs):
            s += f'{timestep}\t{job}\n'
        return s

    def __len__(self):
        """
        returns the number of jobs arrived at this list collector exit point
        """
        return len(self.__jobs)

    def copy(self) -> Exit:
        copy = ListCollectorExit()
        for timestep, job in zip(self.__timesteps, self.__jobs):
            copy.add_job(timestep, job)
        return copy
