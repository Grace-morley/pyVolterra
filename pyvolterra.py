import numpy as np
from abc import ABC
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class Results:
    """
    The results passed back from a integration

    Attributes
    ----------
    x : ndarray
        Returning the user given x array for the user integration.

    y : ndarray
        The array containing the integrated values.

    time_elapsed : float
        The time the intergration took.

    Notes
    -----
    This method is frozen and cannot be altered by the end user.
    """
    x: np.ndarray
    y: np.ndarray
    time_elapsed: float


class Timer:
    """
    A context manager that times the integration.

    Attributes
    ----------
    time_begin : float
        The time that the context manager is entered.

    time_elapsed : float
        The time that the context manager is exited.

    Methods
    -------
    get_time :
        Returns the time elapsed during the integration

    """

    def __init__(self):
        pass

    def __enter__(self):
        self.time_begin = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.time_begin

    def get_time(self):
        """
        Get the time the integration took

        Return
        ------
        elapsed_time : float
            The time that the context manager is exited.
        """
        return self.elapsed_time


class AbstractSolver(ABC):
    """
    The parent class of all pyVolterra solvers

    Attributes
    ----------
    This class has no default attributes and must have them set by methods.

    Methods
    -------
    import_kernel(func, args) :
        Sets the kernel and kernel args for the integration.

    import_g(func, args) :
        Sets the optional G function for the integration.

    set_time_array(t) :
        Sets the array to be used along the x dimension in the integration.

    evaluate() :
        A dummy method included to force inheritance in child classes.

    """

    def __init__(self):
        pass

    def import_kernel(self, func, args):
        """
        Set the K function and args to be used in the integration.

        parameters
        ----------

        func(t_dash, t, args) : Callable
            The function taking three arguments to be used as K in the integration.

        args : list
            A list of args that can be passed to func during the integration

        Raises
        ------
        Exception :
            Raised when func is not a callable function

        Notes
        -----

        Both func and args are set as class attributes a called kernel and kernel_args
        """
        if not callable(func):
            raise Exception('Error: First argument must be a callable function')
        self.kernel = func
        self.kernel_args = args

    def import_g(self, func, args):
        """
        Set the optional G function and args to be used in the integration.

        parameters
        ----------

        func(t_dash, t, args) : Callable
            The function taking three arguments to be used as G in the integration.

        args : list
            A list of args that can be passed to func during the integration

        Raises
        ------
        Exception :
            Raised when func is not a callable function

        Notes
        -----

        Both func and args are set as class attributes a called g and g_args
        """

        if not callable(func):
            raise Exception('Error: First argument must be a callable function')

        self.g = func
        self.g_args = args

    def set_time_array(self, t: np.ndarray):
        self.time_array = t

    def _time_enumerate(self, collection):
        for i, val in enumerate(collection[1:]):
            yield i, val, collection[:i + 1]

    def _evaluate_kernel(self, t_0, t_arr):
        return self.kernel(t_arr, t_0, self.kernel_args)

    def _set_up(self):
        self.h = self.time_array[1] - self.time_array[0]
        try:
            self.g_vals = self.g(self.time_array, self.g_args)
        except AttributeError:
            self.g_vals = np.zeros(len(self.time_array))

    def evaluate(self, initial_value: float):
        """
        A dummy method used to force inheritance to all child classes
        """
        pass


def _evaluate_hidden(time_array, _time_enumerate, _evaluate_kernel, f, g, h):
    for i, t_i, time_arr in _time_enumerate(time_array):
        k = _evaluate_kernel(t_i, time_arr)
        f[i] = (h * (k[0] * f[0] / 2 + np.sum(f[1:i] * k[1:i])) + g[i]) / (1 - k[-1] * h / 2)
    return f


class VolterraSecMarch(AbstractSolver):
    """
    Class for solving Volterra equations of the second kind using the marching method.

    Attributes
    ----------
    This class has no default attributes and must have them set by methods.

    Methods
    -------
    import_kernel(func, args) :
        Sets the kernel and kernel args for the integration.

    import_g(func, args) :
        Sets the optional G function for the integration.

    set_time_array(t) :
        Sets the array to be used along the x dimension in the integration.

    evaluate(initial_value) :
        Sets the initial conditions for the integration and evaluated the integral

    """

    def __init__(self):
        pass

    def evaluate(self, initial_value: float):
        """
        Evaluate the integral over the specified t values for the given K function

        Parameters
        ----------
        initial_value : float
            The initial value for the integration at t[0].

        Return
        ------
        Results : class
            Returns a Results class with the values calculated for each of the given t values

        Notes
        -----
        The set_time_array and the import_kernel methods must be called first to perform the integration and will cause
        crashes otherwise. The import_g method is optional as g is optional and will not course errors if not called.
        """
        with Timer() as int_time:
            self._set_up()
            f = np.zeros(len(self.time_array))
            f[0] = initial_value
            f = _evaluate_hidden(self.time_array, self._time_enumerate, self._evaluate_kernel, f, self.g_vals, self.h)

        return Results(self.time_array, f, int_time.get_time())


class VolterraSecBlcByBlc(AbstractSolver):
    """
    Class for solving Volterra equations of the second kind using the block by block method.

    Attributes
    ----------
    This class has no default attributes and must have them set by methods.

    Methods
    -------
    import_kernel(func, args) :
        Sets the kernel and kernel args for the integration.

    import_g(func, args) :
        Sets the optional G function for the integration.

    set_block_length(N) :
        Sets the number of points to be simultaneously calculated

    set_time_array(t) :
        Sets the array to be used along the x dimension in the integration.

    evaluate(initial_value) :
        Sets the initial conditions for the integration and evaluated the integral
    """

    def __init__(self):
        pass

    def set_block_length(self, N: int):
        """
        Set the number of points to be simultaneously calculated but the block by block calculation

        parameters
        ----------
        N : int
            The number of points to be simultaneously calculated. This value cannot be less then 2.

        Raise
        -----
        Exception :
            Raised when N<=1 as this is a invalid number of points for the algorithm

        """
        if N <= 1:
            raise Exception('Error: block length must be greater them 1')
        self.block_len = N

    def set_time_array(self, t: np.ndarray):
        """
        Set the x dimension array for the integration.

        Parameters
        ----------
        t : ndarray

        Raise
        -----

        Exception :
            An exception can be raised if the set_block_length perimeter is not called before this method

        Exception :
            An exception can be raised if the length of the inputted t is not a integer multiple of N

        Notes
        -----
        The set_block_length method must be called before this method is called or a error will be encountered
        """
        try:
            self.block_len
        except AttributeError:
            raise Exception('Error: No block length defined.')

        if len(t) % self.block_len != 0:
            raise Exception('Error: y values must be a integer multiple of the block length.')

        self.time_array = t

    def _time_enumerate(self, collection):
        for i, val in enumerate(collection[::2]):
            yield i, val, collection[:i + 1]

    def evaluate(self, initial_value: float):
        """
        Evaluate the integral over the specified t values for the given K function using a block by block method

        Parameters
        ----------
        initial_value : float
            The initial value for the integration at t[0].

        Return
        ------
        Results : class
            Returns a Results class with the values calculated for each of the given t values

        Notes
        -----
        The set_time_array, set_block_length, and the import_kernel methods must be called first to perform the
        integration and will cause crashes otherwise. The import_g method is optional as g is optional and will not
        course errors if not called.
        """

        with Timer() as int_time:
            self._set_up()

            self.f = np.asarray([initial_value])

            for i, t_i, time_arr in self._time_enumerate(self.time_array):
                kernel = self._evaluate_kernel(t_i, time_arr)
                self.f = np.append(self.f, _val_at_t(i, t_i, time_arr, kernel, self.f, self.g_vals))

        return Results(self.time_array, self.f, int_time.get_time())

    def _val_at_t(self, i, t_i, trunc_time):

        kernel = self._evaluate_kernel(t_i, trunc_time)
        fi = (self.h * (kernel[0] * self.f[0] / 2 + np.sum(self.f[1:i] * kernel[1:i])) + self.g_vals[i]) / (
                1 - kernel[-1] * self.h / 2)
        return fi
