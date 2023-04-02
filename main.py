import time

import meshio
import numpy as np
import shepard
import linear
from numba import cuda

# крупная сетка
surface = meshio.read("sphere.msh")
# измельченная сетка
better_surface = meshio.read("better_sphere.msh")

device = cuda.get_current_device()

points_surface = []
default_temps = []
for i in range(len(surface.points)):
    points_surface.append([surface.points[i][0], surface.points[i][1], surface.points[i][2]])

default_temps = np.linspace(26.6, 36.6, len(points_surface))

sh = shepard.Shepard2D(points=points_surface,
                       values=default_temps)
ln = linear.Linear(points=points_surface,
                   values=default_temps)

surface_with_data = meshio.Mesh(points=surface.points, cells=surface.cells,
                                point_data={"T": default_temps})

interpolated_temps_linear = []
interpolated_temps_shepard = []
points_better_surface = []
cells_better_surface = []

start = time.time()
for i in range(len(better_surface.points)):
    interpolated_temps_linear.append(ln.interpolate(better_surface.points[i][0],
                                                    better_surface.points[i][1],
                                                    better_surface.points[i][2]))
end = time.time()
print("Elapsed (after compilation) linear interpolation = %s" % (end - start))
start = time.time()
for i in range(len(better_surface.points)):
    interpolated_temps_shepard.append(sh.interpolate(better_surface.points[i][0],
                                                     better_surface.points[i][1],
                                                     better_surface.points[i][2]))
end = time.time()
print("Elapsed (after compilation) shepard interpolation = %s" % (end - start))

better_surface_with_data_linear = meshio.Mesh(better_surface.points, better_surface.cells,
                                              point_data={"T": interpolated_temps_linear})
better_surface_with_data_shepard = meshio.Mesh(better_surface.points, better_surface.cells,
                                               point_data={"T": interpolated_temps_shepard})

surface_with_data.write("surface_1_temps", file_format="gmsh22")
better_surface_with_data_linear.write("surface_2_temps_linear", file_format="gmsh22")
better_surface_with_data_shepard.write("surface_2_temps_shepard", file_format="gmsh22")
