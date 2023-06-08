from numba import cuda
import numpy as np


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:,:], float64[:], float64[:])')
def interpolate_cuda(x, y, z, points, values, interp_values):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= len(x):
        return
    weight = 0.0
    summa = 0.0

    for j in range(len(points)):
        d1 = x[i] - points[j][0]
        d2 = y[i] - points[j][1]
        d3 = z[i] - points[j][2]
        r = d1 * d1 + d2 * d2 + d3 * d3
        if r == 0.0:
            interp_values[i] = values[j]
            return
        w = r ** (-10 / 2)
        weight += w
        summa += w * values[j]

    interp_values[i] = summa / weight
