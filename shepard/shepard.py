class Shepard:
    points: list
    values: list
    p = -7

    def __init__(self, points, values):
        self.points = points
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
