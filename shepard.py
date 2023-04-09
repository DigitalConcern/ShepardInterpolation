from numba import cuda
import numpy as np


class Shepard2D:
    points: list
    cells: np.ndarray
    values: list
    p = -5

    def __init__(self, points, values, cells):
        self.points = points
        self.cells = cells
        self.values = values

    def interpolate(self, x, y, z):
        weight = 0.0
        summa = 0.0

        for i in range(len(self.points) - 1):
            d1 = x - self.points[i][0]
            d2 = y - self.points[i][1]
            d3 = z - self.points[i][2]
            r = d1 * d1 + d2 * d2 + d3 * d3
            if r == 0.0:
                return self.values[i]
            w = r ** (self.p / 2)
            weight += w
            summa += w * self.values[i]
        return summa / weight


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:,:], float64[:], float64[:])')
def interpolate_cuda(x, y, z, points, values, interp_values):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    print(cuda.blockIdx.x)
    if i >= len(x):
        return
    weight = 0.0
    summa = 0.0

    for j in range(len(points) - 1):
        d1 = x[i] - points[j][0]
        d2 = y[i] - points[j][1]
        d3 = z[i] - points[j][2]
        r = d1 * d1 + d2 * d2 + d3 * d3
        if r == 0.0:
            interp_values[i] = values[j]
        w = r ** (-7 / 2)
        weight += w
        summa += w * values[j]

    interp_values[i] = summa / weight
