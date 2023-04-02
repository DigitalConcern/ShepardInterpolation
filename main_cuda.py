import time

import meshio
import numpy as np
from shepard import interpolate_cuda
from numba import cuda

# крупная сетка
surface = meshio.read("sphere.msh")
# измельченная сетка
better_surface = meshio.read("better_sphere.msh")

points_surface = []
default_temps = []
for i in range(len(surface.points)):
    points_surface.append([surface.points[i][0], surface.points[i][1], surface.points[i][2]])

default_temps = np.linspace(26.6, 36.6, len(points_surface))

surface_with_data = meshio.Mesh(points=surface.points, cells=surface.cells,
                                point_data={"T": default_temps})

X = []
Y = []
Z = []
for i in range(len(better_surface.points)):
    X.append(better_surface.points[i][0])
    Y.append(better_surface.points[i][1])
    Z.append(better_surface.points[i][2])

interpolated_temps_empty = np.empty(len(X), dtype=np.float64)

# Подробности об устройстве
device = cuda.get_current_device()

d_x = cuda.to_device(X)
d_y = cuda.to_device(Y)
d_z = cuda.to_device(Z)
d_points = cuda.to_device(points_surface)
d_values = cuda.to_device(default_temps)
d_interpolated_values = cuda.to_device(interpolated_temps_empty)

tpb = device.WARP_SIZE  # blocksize или количество потоков на блок
bpg = int(np.ceil(len(X) / tpb))  # блоков на грид

start = time.time()
interpolate_cuda[bpg, tpb](d_x, d_y, d_z, d_points, d_values, d_interpolated_values)  # вызов ядра
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))

# Перенос вывода с устройства на хост
cuda.synchronize()
interpolated_temps = d_interpolated_values.copy_to_host()

better_surface_with_data = meshio.Mesh(better_surface.points, better_surface.cells,
                                       point_data={"T": interpolated_temps})

surface_with_data.write("surface_1_temps", file_format="gmsh22")
better_surface_with_data.write("surface_2_temps", file_format="gmsh22")
