import time

import meshio
import numpy as np
import shepard
import linear
from numba import cuda

# крупная сетка
surface = meshio.read("m1_1_gmsh.msh")
# измельченная сетка
better_surface = meshio.read("better_m1_1_gmsh.msh")

default_temps = np.linspace(273, 473, len(surface.points))

surface_with_data = meshio.Mesh(points=surface.points, cells=surface.cells,
                                point_data={"T": default_temps})

X = []
Y = []
Z = []
for i in range(len(better_surface.points)):
    X.append(better_surface.points[i][0])
    Y.append(better_surface.points[i][1])
    Z.append(better_surface.points[i][2])

interpolated_temps_shepard_empty = np.empty(len(X), dtype=np.float64)
interpolated_temps_linear_empty = np.empty(len(better_surface.points), dtype=np.float64)

# Подробности об устройстве
device = cuda.get_current_device()

d_x = cuda.to_device(X)
d_y = cuda.to_device(Y)
d_z = cuda.to_device(Z)
d_points = cuda.to_device(surface.points)
d_new_points = cuda.to_device(better_surface.points)
d_cells = cuda.to_device(surface.cells[0].data)
d_values = cuda.to_device(default_temps)
d_interpolated_values_linear = cuda.to_device(interpolated_temps_linear_empty)
d_interpolated_values_shepard = cuda.to_device(interpolated_temps_shepard_empty)

tpb = device.WARP_SIZE  # blocksize или количество потоков на блок
bpg = int(np.ceil(len(X) / tpb))  # блоков на грид

start = time.time()
shepard.interpolate_cuda[bpg, tpb](d_x, d_y, d_z, d_points, d_values, d_interpolated_values_shepard)  # вызов ядра
end = time.time()
print("Elapsed (after compilation) shepard interpolation = %s" % (end - start))

start = time.time()
interpolated_temps_linear = []
for point in better_surface.points:
    interpolated_temps_linear.append(linear.interpolate_cuda(
        point,
        surface.points,
        surface.cells[0].data,
        default_temps,
        device
    ))
    # print(point, interpolated_temps_linear, sep="----->")
end = time.time()
print("Elapsed (after compilation) linear interpolation = %s" % (end - start))

# Перенос вывода с устройства на хост
cuda.synchronize()
interpolated_temps_shepard = d_interpolated_values_shepard.copy_to_host()

better_surface_with_data_shepard = meshio.Mesh(better_surface.points, better_surface.cells,
                                               point_data={"T": interpolated_temps_shepard})

better_surface_with_data_linear = meshio.Mesh(better_surface.points, better_surface.cells,
                                              point_data={"T": interpolated_temps_linear})

surface_with_data.write("m1_1_gmsh_with_data_cuda", file_format="gmsh22")
better_surface_with_data_linear.write("better_m1_1_gmsh_with_data_cuda_linear", file_format="gmsh22")
better_surface_with_data_shepard.write("better_m1_1_gmsh_with_data_cuda_shepard", file_format="gmsh22")
