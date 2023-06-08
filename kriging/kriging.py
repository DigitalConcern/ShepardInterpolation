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
        self.values = values
        self.variogram = skg.Variogram(points, values)

    def interpolate(self, points):
        ok = skgstat.OrdinaryKriging(self.variogram)
        v_values = ok.transform(points)
        return v_values
