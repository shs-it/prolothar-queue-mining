from prolothar_queue_mining.model.exit.exit import Exit
from prolothar_queue_mining.model.job import Job

class DoNothingExit(Exit):
    """
    exit point that does nothing (null interface pattern)
    """

    def add_job(self, timestep: int, job: Job):
        #as the name suggests: do nothing
        pass

    def copy(self) -> Exit:
        return DoNothingExit()