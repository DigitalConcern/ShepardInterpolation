import meshio
import numpy as np
import shepard

# крупная сетка
surface = meshio.read("surface.msh")
# измельченная сетка
better_surface = meshio.read("better_surface.msh")

points_surface = []
default_temps = []
for i in range(len(surface.points)):
    points_surface.append([surface.points[i][0], surface.points[i][1]])

default_temps = np.linspace(26.6, 36.6, len(points_surface))
sh = shepard.Shepard2D(points=points_surface,
                       array_t=default_temps)

surface_with_data = meshio.Mesh(points=surface.points, cells=surface.cells,
                                point_data={"T": default_temps})

interpolated_temps = []
points_better_surface = []
cells_better_surface = []
for i in range(len(better_surface.points)):
    points_better_surface.append([better_surface.points[i][0],
                                  better_surface.points[i][1],
                                  better_surface.points[i][2]])
    interpolated_temps.append(sh.interpolate(better_surface.points[i][0],
                                             better_surface.points[i][1],
                                             better_surface.points[i][2]))

better_surface_with_data = meshio.Mesh(better_surface.points, better_surface.cells,
                                       point_data={"T": interpolated_temps})


surface_with_data.write("surface_1_temps", file_format="gmsh22")
better_surface_with_data.write("surface_2_temps", file_format="gmsh22")
