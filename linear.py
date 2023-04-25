import math
import time

from klampt.math import so3
import numpy as np
from numba import njit, cuda


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
        # start = time.time()
        cells = self.search_cells_with_boxes(x, y, z)
        # print(cells)
        # end = time.time()
        # print("Elapsed (after compilation) search_cells_with_boxes for 1 vertex = %s" % (end - start))

        # start = time.time()
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
            new_point_a = so3.apply(rotation_matrix, point_a)
            new_point_b = so3.apply(rotation_matrix, point_b)
            new_point_c = so3.apply(rotation_matrix, point_c)
            new_point_target = so3.apply(rotation_matrix, p_point_target)

            value_a = self.values[cell[0]]
            value_b = self.values[cell[1]]
            value_c = self.values[cell[2]]

            A = np.array([[new_point_a[0], new_point_b[0], new_point_c[0]],
                          [new_point_a[1], new_point_b[1], new_point_c[1]],
                          [1, 1, 1]])
            B = np.array([new_point_target[0], new_point_target[1], 1])
            try:
                X = np.linalg.solve(A, B)
            except:
                continue
            if round(X[0], 5) < 0.0 or round(X[1], 5) < 0.0 or round(X[2], 5) < 0.0:
                continue
            # end = time.time()
            # print("Elapsed (after compilation) barycentric coordinates for found boxes = %s" % (end - start))
            return X[0] * value_a + X[1] * value_b + X[2] * value_c


@njit
# @cuda.jit('void(float64[:], float64[:,:], int32[:,:], int32[:,:])')
def search_cells_with_boxes_cuda(new_point, points, cells, result_cells):
    counter = 0
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # if i >= len(cells):
    #     return
    for i in range(len(cells)):
        id_point_1 = cells[i][0]
        id_point_2 = cells[i][1]
        id_point_3 = cells[i][2]
        # Поиск минимального X для треугольника
        min_x = points[id_point_1][0]
        if min_x > points[id_point_2][0]:
            min_x = points[id_point_2][0]
        if min_x > points[id_point_3][0]:
            min_x = points[id_point_3][0]
        # Поиск максимального X для треугольника
        max_x = points[id_point_1][0]
        if max_x < points[id_point_2][0]:
            max_x = points[id_point_2][0]
        if max_x < points[id_point_3][0]:
            max_x = points[id_point_3][0]
        # Поиск минимального Y для треугольника
        min_y = points[id_point_1][1]
        if min_y > points[id_point_2][1]:
            min_y = points[id_point_2][1]
        if min_y > points[id_point_3][1]:
            min_y = points[id_point_3][1]
        # Поиск максимального Y для треугольника
        max_y = points[id_point_1][1]
        if max_y < points[id_point_2][1]:
            max_y = points[id_point_2][1]
        if max_y < points[id_point_3][1]:
            max_y = points[id_point_3][1]
        # Поиск минимального Z для треугольника
        min_z = points[id_point_1][2]
        if min_z > points[id_point_2][2]:
            min_z = points[id_point_2][2]
        if min_z > points[id_point_3][2]:
            min_z = points[id_point_3][2]
        # Поиск максимального Z для треугольника
        max_z = points[id_point_1][2]
        if max_z < points[id_point_2][2]:
            max_z = points[id_point_2][2]
        if max_z < points[id_point_3][2]:
            max_z = points[id_point_3][2]
        if min_x <= new_point[0] <= max_x:
            if min_y <= new_point[1] <= max_y:
                if min_z <= new_point[2] <= max_z:
                    result_cells[counter][:3] = cells[i][:3]
                    # for j in range(len(cells[i])):
                    #     result_cells[counter][j] = cells[i][j]
                    counter += 1
    return result_cells


def interpolate_cuda(new_point, points, cells, values, device):
    start = time.time()
    # d_points = cuda.to_device(points)
    # d_new_point = cuda.to_device(new_point)
    # d_cells = cuda.to_device(cells)

    result_cells_empty = np.array([[0, 0, 0]] * len(cells), dtype=np.int32)
    # d_result_cells = cuda.to_device(result_cells_empty)

    # tpb = device.WARP_SIZE
    # bpg = int(np.ceil(len(points) / tpb))

    result_cells = \
        search_cells_with_boxes_cuda(new_point,
                                     points,
                                     cells,
                                     result_cells_empty,
                                     )

    # result_cells = [x for x in result_cells if sum(x) != 0]
    # cuda.synchronize()
    # result_cells = d_result_cells.copy_to_host()
    # print(result_cells)

    end = time.time()
    print("Elapsed (after compilation) search_cells_with_boxes for 1 vertex = %s" % (end - start))

    # start = time.time()
    for cell in result_cells:
        if sum(cell) != 0:
            point_a = points[cell[0]]
            point_b = points[cell[1]]
            point_c = points[cell[2]]

            vector_1 = point_b - point_a
            vector_2 = point_c - point_a

            surface_vector = np.cross(vector_1, vector_2)

            vector_norm = surface_vector[0] * surface_vector[0] + surface_vector[1] * surface_vector[1] + \
                          surface_vector[2] * surface_vector[2]
            normal_vector = surface_vector / np.sqrt(vector_norm)

            # Вычисляем проекцию точки
            point_target = np.array(new_point, dtype=np.float64)
            point_from_point_in_plane = point_target - point_a
            proj_onto_normal_vector = np.dot(point_from_point_in_plane, normal_vector)
            proj_onto_plane = (point_from_point_in_plane - proj_onto_normal_vector * normal_vector)
            p_point_target = point_a + proj_onto_plane

            # Вычисляем ось и угол поворота треугольника
            rotation_axis = np.cross(normal_vector, np.array([0, 0, 1.0]))
            angle_n_z = math.acos(np.dot(normal_vector, np.array([0, 0, 1.0])))

            # Вычисляем матрицу поворота и координаты новых точек
            rotation_matrix = so3.rotation(rotation_axis, angle_n_z)
            new_point_a = so3.apply(rotation_matrix, point_a)
            new_point_b = so3.apply(rotation_matrix, point_b)
            new_point_c = so3.apply(rotation_matrix, point_c)
            new_point_target = so3.apply(rotation_matrix, p_point_target)

            value_a = values[cell[0]]
            value_b = values[cell[1]]
            value_c = values[cell[2]]

            A = np.array([[new_point_a[0], new_point_b[0], new_point_c[0]],
                          [new_point_a[1], new_point_b[1], new_point_c[1]],
                          [1, 1, 1]])
            B = np.array([new_point_target[0], new_point_target[1], 1])

            try:
                X2 = np.linalg.solve(A, B)
            except:
                continue

            if round(X2[0], 5) < 0.0 or round(X2[1], 5) < 0.0 or round(X2[2], 5) < 0.0:
                continue

            # end = time.time()
            # print("Elapsed (after compilation) barycentric coordinates for found boxes = %s" % (end - start))
            return X2[0] * value_a + X2[1] * value_b + X2[2] * value_c
