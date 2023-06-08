import math
import time

from klampt.math import so3
import numpy as np


def calculate_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class Barycentric:
    points: list
    cells: np.ndarray
    values: list

    X: list
    Y: list
    Z: list

    def __init__(self, points, cells, values):
        self.points = points
        self.cells = cells
        self.values = values

        self.X = [point[0] for point in points]
        self.Y = [point[1] for point in points]
        self.Z = [point[2] for point in points]

    def search_cells_with_boxes(self, x, y, z):
        results = []
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
                results.append(cell)

        return results

    def search_cells_with_point(self, index):
        results = []
        for cell in self.cells:
            for point_index in cell:
                if index == point_index:
                    results.append(cell)
        return results

    def search_nearest_neighbour(self, x, y, z):
        d1 = x - self.points[0][0]
        d2 = y - self.points[0][1]
        d3 = z - self.points[0][2]
        r_min = d1 * d1 + d2 * d2 + d3 * d3
        index_min = 0

        for i in range(len(self.points)):
            d1 = x - self.points[i][0]
            d2 = y - self.points[i][1]
            d3 = z - self.points[i][2]
            r = d1 * d1 + d2 * d2 + d3 * d3

            if r < r_min:
                r_min = r
                index_min = i

        return index_min

    def interpolate(self, x, y, z):
        # cells = self.search_cells_with_boxes(x, y, z)

        index = self.search_nearest_neighbour(x, y, z)

        start = time.time()
        cells = self.search_cells_with_point(index)
        end = time.time()
        print("Elapsed (after compilation) linear interpolation = %s" % (end - start))

        results = []
        for cell in cells:
            point_a = self.points[cell[0]]
            point_b = self.points[cell[1]]
            point_c = self.points[cell[2]]

            vector_1 = point_b - point_a
            vector_2 = point_c - point_a

            surface_vector = np.cross(vector_1, vector_2)

            vector_norm = surface_vector[0] * surface_vector[0] + surface_vector[1] * surface_vector[1] + \
                          surface_vector[2] * surface_vector[2]
            normal_vector = surface_vector / np.sqrt(vector_norm)

            # Вычисляем проекцию точки
            point_target = np.array([x, y, z], dtype=np.float64)
            point_from_point_in_plane = point_target - point_a
            proj_onto_normal_vector = np.dot(point_from_point_in_plane, normal_vector)
            proj_onto_plane = (point_from_point_in_plane - proj_onto_normal_vector * normal_vector)
            p_point_target = point_a + proj_onto_plane

            # Вычисляем ось и угол поворота треугольника
            rotation_axis = np.cross(normal_vector, np.array([0, 0, 1.0]))
            angle_n_z = math.acos(np.dot(normal_vector, np.array([0, 0, 1.0])))

            # Вычисляем матрицу поворота и координаты новых точек
            rotation_matrix = so3.rotation(rotation_axis, angle_n_z)
            new_point_a = np.array(so3.apply(rotation_matrix, point_a))
            new_point_b = np.array(so3.apply(rotation_matrix, point_b))
            new_point_c = np.array(so3.apply(rotation_matrix, point_c))
            new_point_target = np.array(so3.apply(rotation_matrix, p_point_target))

            value_a = self.values[cell[0]]
            value_b = self.values[cell[1]]
            value_c = self.values[cell[2]]

            A = np.array([[new_point_a[0], new_point_b[0], new_point_c[0]],
                          [new_point_a[1], new_point_b[1], new_point_c[1]],
                          [1, 1, 1]])
            B = np.array([new_point_target[0], new_point_target[1], 1])
            result = np.linalg.solve(A, B)

            if result[0] < 0.0 or result[1] < 0.0 or result[2] < 0.0:
                continue

            results.append(result[0] * value_a + result[1] * value_b + result[2] * value_c)

        if not results:
            return self.values[index]
        else:
            # проверка массива на NaN
            results_no_nan = []
            for x in results:
                if not np.isnan(float(x)):
                    results_no_nan.append(x)
            if not results_no_nan or len(results_no_nan) > 1:
                return self.values[index]
            else:
                return results_no_nan[0]