class Error(Exception):
    """
    Base class for package errors
    """
    pass


class MethodOrderError(Error):
    """
    A Error raised when methods are called in the wrong order and the integrator cannot function
    """

    def __init__(self, atribute):
        print('Error: {} not defined before integrator called'.format(atribute))
