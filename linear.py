import math
import numpy as np


class Linear:
    points: list
    values: list

    def __init__(self, points, values):
        self.points = points
        self.values = values

    def search(self, x, y, z):
        nearest_indexes = []

        least_distance = -1
        for i in range(len(self.points) - 1):
            # Euclidean distance formula
            distance = math.sqrt(
                ((x - self.points[i][0]) ** 2) + ((y - self.points[i][1]) ** 2) + ((z - self.points[i][2]) ** 2))
            # Finding the least distance value
            if distance == 0.0:
                continue
            if distance <= least_distance or least_distance == -1:
                least_distance = distance
                nearest_indexes.insert(0, i)
            else:
                nearest_indexes.append(i)

        c_list = nearest_indexes[:3]
        return c_list[0], c_list[1], c_list[2]

    def interpolate(self, x, y, z):
        indexes = self.search(x, y, z)
        distances = []

        total_distance = 0.0
        for i in indexes:
            distance = math.sqrt(
                ((x - self.points[i][0]) ** 2) + ((y - self.points[i][1]) ** 2) + ((z - self.points[i][2]) ** 2))
            distances.append(distance)
            total_distance += distance

        summa = 0.0
        for i in range(len(distances) - 1):
            summa += self.values[indexes[i]] * distances[i] / total_distance

        return summa
