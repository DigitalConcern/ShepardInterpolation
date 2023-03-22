class Shepard2D:
    points: list
    temp: list
    p = -7

    def __init__(self, points, array_t):
        self.points = points
        self.temp = array_t

    def interpolate(self, x, y, z):
        weight = 0.0
        summa = 0.0
        for i in range(len(self.points) - 1):
            d1 = x - self.points[i][0]
            d2 = y - self.points[i][1]
            d3 = z - self.points[i][2]
            r = d1 * d1 + d2 * d2 + d3 * d3
            if r == 0.0:
                return self.temp[i]
            w = pow(r, self.p / 2)
            weight += w
            summa += w * self.temp[i]
        return summa / weight
