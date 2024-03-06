cdef class Job:
    """
    job model, usually a customer or a product
    """
    def __init__(self, str job_id, dict features = None):
        """
        creates a new job instance

        Parameters
        ----------
        job_id : str
            unique identifier of the job
        features : dict
            additional attributes/features (name, value). the features can change
            over time.
        """
        self.job_id = job_id
        if features is not None:
            self.features = features
        else:
            self.features = {}

    def __hash__(self):
        return hash(self.job_id)

    def __eq__(self, other):
        return self.job_id == other.job_id and self.features == other.features

    def __repr__(self):
        return f'Job({self.job_id}, {self.features})'
