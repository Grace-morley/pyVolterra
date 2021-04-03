import pyvolterra as pv
from matplotlib import pyplot as plt
import numpy as np
import unittest
from packagerrors import *


class TestVolterraSecond(unittest.TestCase):

    def test_VolterraMartch1(self):
        t = np.linspace(0, 10, 50001)

        def kernel(t_dash, t_main, args):
            return t_main - t_dash

        def G(t_main, args):
            return t_main

        ## set up the equation
        test_class = pv.VolterraSecMarch()
        test_class.set_time_array(t)
        test_class.import_kernel(kernel, [])
        test_class.import_g(G, [])
        results = test_class.evaluate(0)
        #
        psi_2 = np.sinh(t)
        psi_3 = (results.y[1:] - psi_2[1:]) / psi_2[1:]

        self.assertIsInstance(results, pv.Results)
        self.assertTrue(all([True if i <= 10 ** -8 else False for i in psi_3]))
        self.assertTrue(True if results.time_elapsed <= 20 else False)

    def test_VolterraMartch2(self):
        t = np.linspace(0, 10, 50001)

        def kernel(t_dash, t, args):
            return (1 + t) / (1 + t_dash)

        def G(t, args):
            return args[0] - t - args[1] * t ** 2 + t ** 3 * args[2]

        test_class = pv.VolterraSecMarch()
        test_class.set_time_array(t)
        test_class.import_kernel(kernel, [])
        test_class.import_g(G, [1, 3 / 2, 1 / 2])
        results = test_class.evaluate(1)

        self.assertTrue(True if results.time_elapsed <= 20 else False)


if __name__ == '__main__':
    unittest.main()