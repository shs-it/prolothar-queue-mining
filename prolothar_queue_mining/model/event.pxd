from prolothar_queue_mining.model.environment cimport Environment

cdef class Event:
    cdef public int time
    cdef public int prio

    cpdef execute(self, Environment environment)