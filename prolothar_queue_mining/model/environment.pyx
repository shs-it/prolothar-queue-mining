from typing import Callable

cdef class Environment:

    def __init__(self, verbose: bool = False, log_function: Callable[[str],None] = print):
        self.__event_queue = EventQueue()
        self.__current_time = 0
        self.__log_function = log_function
        self.__verbose = verbose

    cpdef int get_current_time(self):
        return self.__current_time

    cpdef bint has_open_events(self):
        return self.__event_queue.is_non_empty()

    cpdef schedule_event(self, Event event):
        if event.time < self.__current_time:
            raise ValueError(f'cannot schedule past event "{event}". {event.time} < {self.__current_time}')
        self.__event_queue.enqueue(event)

    cpdef run_timesteps(self, int timesteps):
        cdef int start_time = self.__current_time
        cdef int end_time = start_time + timesteps
        cdef Event next_event
        while self.__event_queue.is_non_empty():
            next_event = self.__event_queue.pop()
            if next_event.time > end_time:
                self.__event_queue.enqueue(next_event)
                break
            self.__current_time = next_event.time
            if self.__verbose:
                self.__log_function(str(next_event))
            next_event.execute(self)
        self.__current_time = end_time

    cpdef run_until(self, condition: Callable[[], bool]):
        cdef Event next_event
        while not condition():
            next_event = self.__event_queue.pop()
            self.__current_time = next_event.time
            if self.__verbose:
                self.__log_function(str(next_event))
            next_event.execute(self)

    cpdef run_until_event_queue_is_empty(self):
        cdef Event next_event
        while self.__event_queue.is_non_empty():
            next_event = self.__event_queue.pop()
            self.__current_time = next_event.time
            if self.__verbose:
                self.__log_function(str(next_event))
            next_event.execute(self)

cdef class EventQueue:
    """
    https://github.com/kilian-gebhardt/MinMaxHeap/blob/master/pyminmaxheap.py
    """

    def __init__(self):
        self.event_list = []
        self.size = 0

    def __len__(self):
        return self.size

    cdef bint is_non_empty(self):
        return self.size != 0

    cdef enqueue(self, Event event):
        if len(self.event_list) < self.size + 1:
            self.event_list.append(event)
        insert_event(self.event_list, event, self.size)
        self.size += 1

    cdef Event pop(self):
        cdef Event event = removemin(self.event_list, self.size)
        self.size -= 1
        return event

cdef insert_event(list array, Event event, unsigned int size):
    array[size] = event
    bubbleup(array, size)

cdef bubbleup(list array, unsigned int i):
    if level(i) % 2 == 0:  # min level
        if i > 0 and array[i] > array[(i-1) // 2]:
            array[i], array[(i-1) // 2] = array[(i-1)//2], array[i]
            bubbleupmax(array, (i-1)//2)
        else:
            bubbleupmin(array, i)
    else:  # max level
        if i > 0 and array[i] < array[(i-1) // 2]:
            array[i], array[(i-1) // 2] = array[(i-1) // 2], array[i]
            bubbleupmin(array, (i-1)//2)
        else:
            bubbleupmax(array, i)

cdef bubbleupmin(list array, unsigned int i):
    while i > 2:
        if array[i] < array[(i-3) // 4]:
            array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
            i = (i-3) // 4
        else:
            return

cdef bubbleupmax(list array, unsigned int i):
    while i > 2:
        if array[i] > array[(i-3) // 4]:
            array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
            i = (i-3) // 4
        else:
            return

cdef unsigned int level(unsigned int n):
    cdef int count=0,i;
    n += 1
    for i in range(32):
        if (1 << i) & n:
            count = i;
    return count;

cdef trickledown(list array, unsigned int i, unsigned int size):
    if level(i) % 2 == 0:  # min level
        trickledownmin(array, i, size)
    else:
        trickledownmax(array, i, size)

cdef trickledownmin(list array, unsigned int i, unsigned int size):
    cdef unsigned int m
    while size > i * 2 + 1:  # i has children
        m = i * 2 + 1
        if m + 1 < size and array[m+1] < array[m]:
            m += 1
        child = True
        for j in range(i*4+3, min(i*4+7, size)):
            if array[j] < array[m]:
                m = j
                child = False

        if child:
            if array[m] < array[i]:
                array[i], array[m] = array[m], array[i]
            break
        else:
            if array[m] < array[i]:
                array[m], array[i] = array[i], array[m]
                if array[m] > array[(m-1) // 2]:
                    array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
                i = m
            else:
                break

cdef trickledownmax(list array, unsigned int i, unsigned int size):
    cdef unsigned int m
    while size > i * 2 + 1:  # i has children
        m = i * 2 + 1
        if i * 2 + 2 < size and array[i*2+2] > array[m]:
            m = i*2+2
        child = True
        for j in range(i*4+3, min(i*4+7, size)):
            if array[j] > array[m]:
                m = j
                child = False

        if child:
            if array[m] > array[i]:
                array[i], array[m] = array[m], array[i]
            break
        else:
            if array[m] > array[i]:
                array[m], array[i] = array[i], array[m]
                if array[m] < array[(m-1) // 2]:
                    array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
                i = m
            else:
                break

cdef Event removemin(list array, unsigned int size):
    cdef Event event = array[0]
    array[0] = array[size-1]
    trickledown(array, 0, size - 1)
    return event


