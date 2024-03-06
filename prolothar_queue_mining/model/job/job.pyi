class Job:
    """
    job model, usually a customer or a product
    """
    job_id: str
    features: dict[str, int|float|str]

    def __init__(self, job_id: str, features: dict[str, int|float|str]|None = None):
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
