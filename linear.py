import math
import numpy as np
import numba as nb


def calculate_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class Linear:
    points: list
    cells: np.ndarray
    values: list

    def __init__(self, points, cells, values):
        self.points = points
        self.cells = cells
        self.values = values

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

        for cell in cells:
            point_a = self.points[cell[0]]
            point_b = self.points[cell[1]]
            point_c = self.points[cell[2]]
            value_a = self.values[cell[0]]
            value_b = self.values[cell[1]]
            value_c = self.values[cell[2]]

            A = np.array([[point_a[0], point_b[0], point_c[0]], [point_a[1], point_b[1], point_c[1]], [point_a[2], point_b[2], point_c[2]]])
            B = np.array([x, y, z])
            X2 = np.linalg.solve(A, B)
            if round(X2[0] + X2[1] + X2[2], 5) > 1.0 or round(X2[0], 5) < 0 or round(X2[1], 5) < 0 or round(X2[2], 5) < 0:
                continue
            return X2[0] * value_a + X2[1] * value_b + X2[2] * value_c
