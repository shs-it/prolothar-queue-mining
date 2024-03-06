cdef class Job:
    """
    job model, usually a customer or a product
    """
    cdef public str job_id
    cdef public dict features
