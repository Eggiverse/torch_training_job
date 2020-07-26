class StepCounter:
    """
    A counter for counting iterations and epochs
    """
    def __init__(self, start=0):
        self.last_step = start - 1

    def __next__(self):
        self.last_step += 1
        return self.last_step

    def __iter__(self):
        return self
