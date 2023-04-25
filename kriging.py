import math

import skgstat
from numba import cuda
import numpy as np
import skgstat as skg


class Krige:
    points: list
    cells: np.ndarray
    values: list
    variogram: skg.Variogram

    def __init__(self, points, values, cells):
        self.points = points
        self.cells = cells
        self.values = values
        self.variogram = skg.Variogram(points, values)

    def search_cells_with_boxes(self, x, y, z):
        cells = []
        for cell in self.cells:
            # Поиск минимального X для треугольника
            min_x = min(self.points[cell[0]][0], self.points[cell[1]][0], self.points[cell[2]][0])
            # Поиск максимального X для треугольника
            max_x = max(self.points[cell[0]][0], self.points[cell[1]][0], self.points[cell[2]][0])
            # Поиск минимального Y для треугольника
            min_y = min(self.points[cell[0]][1], self.points[cell[1]][1], self.points[cell[2]][1])
            # Поиск максимального Y для треугольника
            max_y = max(self.points[cell[0]][1], self.points[cell[1]][1], self.points[cell[2]][1])
            # Поиск минимального Z для треугольника
            min_z = min(self.points[cell[0]][2], self.points[cell[1]][2], self.points[cell[2]][2])
            # Поиск максимального Z для треугольника
            max_z = max(self.points[cell[0]][2], self.points[cell[1]][2], self.points[cell[2]][2])

            if min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z:
                cells.append(cell)

        return cells

    def interpolate(self, x, y, z):
        cells = self.search_cells_with_boxes(x, y, z)
        # print(self.variogram.distance_matrix)

        for cell in cells:
            point_a = self.points[cell[0]]
            point_b = self.points[cell[1]]
            point_c = self.points[cell[2]]

            point_target = np.array([x, y, z], dtype=np.float64)

            ab = math.sqrt(sum(coord * coord for coord in point_a - point_b))
            ac = math.sqrt(sum(coord * coord for coord in point_a - point_c))
            bc = math.sqrt(sum(coord * coord for coord in point_b - point_c))
            a_target = math.sqrt(sum(coord * coord for coord in point_target - point_a))
            b_target = math.sqrt(sum(coord * coord for coord in point_target - point_b))
            c_target = math.sqrt(sum(coord * coord for coord in point_target - point_c))

            ok = skgstat.OrdinaryKriging(self.variogram, min_points=4, max_points=4)
            v_values = ok.transform(np.array([x, point_a[0], point_b[0], point_c[0]]), np.array([y, point_a[1], point_b[1], point_c[1]]), np.array([z, point_a[2], point_b[2], point_c[2]]))
            print(v_values)
            # A = np.array[[], [], [], [1, 1, 1, 1]]


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
        w = r ** (-4 / 2)
        weight += w
        summa += w * values[j]

    interp_values[i] = summa / weight
