from typing import Callable

from prolothar_queue_mining.model.event cimport Event

cdef class EventQueue:
    cdef list event_list
    cdef unsigned int size

    cdef enqueue(self, Event event)
    cdef Event pop(self)
    cdef bint is_non_empty(self)

cdef class Environment:

    cdef EventQueue __event_queue
    cdef int __current_time
    cdef __log_function
    cdef bint __verbose

    cpdef bint has_open_events(self)
    cpdef run_timesteps(self, int timesteps)
    cpdef run_until(self, condition: Callable[[], bool])
    cpdef run_until_event_queue_is_empty(self)
    cpdef schedule_event(self, Event event)
    cpdef int get_current_time(self)