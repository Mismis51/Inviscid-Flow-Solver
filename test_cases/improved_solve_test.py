import src
import numpy as np
import matplotlib.pyplot as plt
from time import time

n = 128
geometry = src.geometry(nb_vertex=n)
# geometry.load_naca(.00, .4, .12)
geometry.load_txt('examples/plate.dat')
src.plot_foil(geometry.vertex)
solver = src.linear_vortex_solver(geometry)

alpha = 10
geometry.set_angle_deg(-alpha)
rotated_vertex = geometry.get_rotated_vertex()

start = time()
gammas = solver.solve(-geometry.angle)
print(f'It took {time() - start} seconds to compute gammas')
print(f'Maximum gamma : {np.max(gammas)}')

plt.scatter(rotated_vertex[:, 0], 1 - gammas**2)
plt.waitforbuttonpress()
plt.clf()

cl, cd, cm = src.compute_coefficients(geometry, gammas)
print(f'Cl is : {cl}')
print(f'Expected Cl for cylinder is {4 * np.pi * np.sin(alpha * np.pi / 180)}')
print(f'Expected Cl for thin foil is {2 * np.pi * alpha * np.pi / 180}')

flows = [src.flows.freestream(1)]
# flows.append(src.flows.linear_vortex(gammas[:-1], 
#                                  gammas[1:], 
#                                  rotated_vertex[:-1], 
#                                  rotated_vertex[1:]))

# Doing it like this is faster. Probably because we have less broadcasting
N = len(gammas)
for i in range(N):
    p1 = rotated_vertex[i]
    p2 = rotated_vertex[(i + 1) % N]

    gamma0 = gammas[i]
    gamma1 = gammas[(i+1) % N]
    flows.append(src.flows.linear_vortex(gamma0, gamma1, p1, p2))

nx, ny = (300, 300)
x = np.linspace(-.5, 1.5, nx)
y = np.linspace(-1, 1, ny)
x_grid, y_grid = np.meshgrid(x, y)

start = time()
result = src.compute_streamlines(flows, x_grid, y_grid)
print(f'Time for streamline computation is {time() - start}s')
src.plot_heightmap(result, x_grid, y_grid, foil=rotated_vertex, levels=None)

u, v = src.compute_velocities(flows, x_grid, y_grid)
src.plot_velmap(u, v, x_grid, y_grid)