class Error(Exception):
    """
    Base class for package errors
    """

    def __str__(self):
        return self.message


class MethodOrderError(Error):
    """
    A Error raised when methods are called in the wrong order and the integrator cannot function
    """

    def __init__(self,*args):
        if args:
            self.message = '{} not defined before integrator called'.format(args[0])
        else:
            self.message = 'Methods called out of order'


class BlockLengthError(Error):
    def __init__(self):
        self.message = 'block length must be greater then 1'

class TimeLenthError(Error):
    def __init__(self):
        self.message = 'y values must be a integer multiple of the block length'