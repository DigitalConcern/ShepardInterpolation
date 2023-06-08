import time

import meshio
import numpy as np
from shepard import shepard
from linear import linear
from kriging import kriging

# крупная сетка
surface = meshio.read("resources/new_mesh_solution2.msh")
# измельченная сетка
better_surface = meshio.read("resources/new_mesh2.msh")

# default_temps = np.linspace(273, 473, len(surface.points))

sh = shepard.Shepard(points=np.array(surface.points, dtype=np.float64),
                     values=surface.point_data["temperature"]) #surface.point_data["temperature"]

ln = linear.Barycentric(points=np.array(surface.points, dtype=np.float64),
                        cells=surface.cells[0].data,
                        values=surface.point_data["temperature"])

kg = kriging.Krige(points=np.array(surface.points, dtype=np.float64),
                   cells=surface.cells[0].data,
                   values=surface.point_data["temperature"])

# surface_with_data = meshio.Mesh(points=surface.points, cells=surface.cells,
#                                 point_data={"T": default_temps})

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

start = time.time()
interpolated_temps_kriging = kg.interpolate(better_surface.points)
end = time.time()
print("Elapsed (after compilation) kriging interpolation = %s" % (end - start))

better_surface_with_data_linear = meshio.Mesh(better_surface.points, better_surface.cells,
                                              point_data={"T": interpolated_temps_linear})
better_surface_with_data_shepard = meshio.Mesh(better_surface.points, better_surface.cells,
                                               point_data={"T": interpolated_temps_shepard})
better_surface_with_data_kriging = meshio.Mesh(better_surface.points, better_surface.cells,
                                               point_data={"T": interpolated_temps_kriging})

# surface_with_data.write("m1_1_gmsh_with_data", file_format="gmsh22")
better_surface_with_data_linear.write("mesh2_linear", file_format="gmsh22")
better_surface_with_data_shepard.write("mesh2_shepard", file_format="gmsh22")
better_surface_with_data_kriging.write("mesh2_kriging", file_format="gmsh22")
