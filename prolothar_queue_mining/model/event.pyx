cdef class Event:
    """
    scheduled event of our discrete event simulation
    """

    def __init__(self, int time, int prio = 0):
        self.time = time
        self.prio = prio

    cpdef execute(self, Environment environment):
        """ executes this event """
        raise NotImplementedError('must be implemented by subclass')

    def __lt__(self, other):
        cdef Event casted_other = <Event>other
        return self.time < casted_other.time or (
            self.time == casted_other.time
            and self.prio < casted_other.prio
        )
