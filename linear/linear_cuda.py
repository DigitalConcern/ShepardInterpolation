import numpy as np
import math
import time

from klampt.math import so3
from numba import njit, cuda


def interpolate_cuda(new_point, points, cells, values, index, result_cells):
    # start = time.time()
    # result_cells_empty = np.array(np.array([[0, 0, 0]] * len(cells), dtype=np.int32) * len())

    # result_cells = search_cells_with_point_cuda(index, cells, result_cells_empty)
    # end = time.time()
    # print("Elapsed (after compilation) linear interpolation = %s" % (end - start))
    # search_cells_with_boxes_cuda(new_point,
    #                              points,
    #                              cells,
    #                              result_cells_empty,
    #                              )

    results = []
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

            X2 = np.linalg.solve(A, B)

            if X2[0] < 0.0 or X2[1] < 0.0 or X2[2] < 0.0:
                continue

            # results.append(X2[0] * value_a + X2[1] * value_b + X2[2] * value_c)
            return X2[0] * value_a + X2[1] * value_b + X2[2] * value_c

    # if not results:
    #     return values[index]
    # else:
    #     # проверка массива на NaN
    #     results_no_nan = []
    #     for x in results:
    #         if not np.isnan(float(x)):
    #             results_no_nan.append(x)
    #     if not results_no_nan or len(results_no_nan) > 1:
    #         return values[index]
    #     else:
    #         return results_no_nan[0]


@njit
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
                    counter += 1
    return result_cells


# @cuda.jit('void(int32[:], int32[:,:], int32[:,:])')
# def search_cells_with_point_cuda(indexes, cells, result_cells):
#     i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     if i >= len(cells):
#         return
#
#     for j in range(len(cells)):
#         for k in range(3):
#             if cells[j][k] == indexes[i]:
#                 for m in range(3):
#                     result_cells[i][j][m] = cells[j][m]


@njit
def search_cells_with_point_cuda(index, cells):
    result_cells = np.array([[0, 0, 0]] * len(cells), dtype=np.int32)
    for i in range(len(cells)):
        for j in range(3):
            if cells[i][j] == index:
                result_cells[i][:3] = cells[i][:3]
    return result_cells


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:,:], int64[:])')
def search_nearest_neighbour_cuda(x, y, z, points, indexes):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= len(x):
        return

    r_min = 100
    index_min = 0

    for j in range(len(points)):
        d1 = x[i] - points[j][0]
        d2 = y[i] - points[j][1]
        d3 = z[i] - points[j][2]
        r = d1 * d1 + d2 * d2 + d3 * d3

        if r < r_min:
            r_min = r
            index_min = j

    indexes[i] = index_min
