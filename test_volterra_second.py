import pyvolterra as pv
from matplotlib import pyplot as plt
import numpy as np
from numba import jit

# Test 1

# solution to equation 1: sinh(x) for x>=0
t = np.linspace(0, 10, 100001)

def kernal(t_dash, t, args):
    return t - t_dash

def G(t, args):
    return t


## set up the equation
test_class = pv.VolterraSecMarch()
test_class.set_time_array(t)
test_class.import_kernel(kernal, [])
test_class.import_g(G, [])
results = test_class.evaluate(0)
#
psi_2 = np.sinh(t)
psi_3 = (results.y - psi_2) / psi_2
plt.plot(t, results.y)
print('time elapsed:{}'.format(results.time_elapsed))
plt.plot(t, psi_2)
plt.show()

## test 2
# t = np.linspace(0, 10, 50001)
#
#
# def kernal(t_dash, t, args):
#     return (1 + t) / (1 + t_dash)
#
#
# def G(t, args):
#     return args[0] - t - args[1] * t ** 2 + t ** 3 * args[2]
#
#
# ## set up the equation
# test_class = pv.VolterraSecMarch()
# test_class.set_time_array(t)
# test_class.import_kernel(kernal, [])
# test_class.import_g(G, [1, 3 / 2, 1 / 2])
# results = test_class.evaluate(1)
#
# plt.plot(t, results.y)
# print('time elapsed:{}'.format(results.time_elapsed))
# plt.plot(t, 1 - t ** 2)
# plt.show()
