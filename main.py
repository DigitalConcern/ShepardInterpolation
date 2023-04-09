import time

import meshio
import numpy as np
import shepard
import linear

# крупная сетка
surface = meshio.read("sphere.msh")
# измельченная сетка
better_surface = meshio.read("better_sphere.msh")

default_temps = np.linspace(26.6, 36.6, len(surface.points))

sh = shepard.Shepard2D(points=surface.points,
                       cells=surface.cells[0].data,
                       values=default_temps)
ln = linear.Linear(points=surface.points,
                   cells=surface.cells[0].data,
                   values=default_temps)

surface_with_data = meshio.Mesh(points=surface.points, cells=surface.cells,
                                point_data={"T": default_temps})

interpolated_temps_linear = []
interpolated_temps_shepard = []

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

surface_with_data.write("sphere_with_data", file_format="gmsh22")
better_surface_with_data_linear.write("better_sphere_with_data_linear", file_format="gmsh22")
better_surface_with_data_shepard.write("better_sphere_with_data_shepard", file_format="gmsh22")
