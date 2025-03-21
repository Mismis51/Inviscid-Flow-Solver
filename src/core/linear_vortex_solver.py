import numpy as np
from ..flows import linear_vortex

class linear_vortex_solver:
    def __init__(self, geometry):
        self.geometry = geometry
        self._create_normals()
        self.RHS = self._create_RHS_matrix()

    def _create_normals(self):
        normals = self.geometry.normal
        n_x, n_y = normals[:-1, 0], normals[:-1, 1]
        self.nx = n_x
        self.ny = n_y

    def _create_RHS_matrix(self):
        vertex = self.geometry.vertex
        centers = self.geometry.center

        N = len(vertex)
        A = np.zeros((N, N))

        # Panel endpoints
        p1, p2 = vertex[:-1], vertex[1:]

        # Panel centers
        x, y = centers[:-1, 0], centers[:-1, 1]

        # Right influence (gamma_a = 1, gamma_b = 0)
        right_panels = linear_vortex(np.ones(N - 1), np.zeros(N - 1), p1, p2)
        u_vec, v_vec = right_panels.velocity(x, y)
        A_right = u_vec * self.nx[:, np.newaxis] + v_vec * self.ny[:, np.newaxis]
        A[:-1, :-1] += A_right

        # Left influence (gamma_a = 0, gamma_b = 1)
        left_panels = linear_vortex(np.zeros(N - 1), np.ones(N - 1), p1, p2)
        u_vec, v_vec = left_panels.velocity(x, y)
        A_left = u_vec * self.nx[:, np.newaxis] + v_vec * self.ny[:, np.newaxis]
        A[:-1, 1:] += A_left

        # Kutta condition
        A[-1, 0] = 1.0
        A[-1, -1] = 1.0
        return A
    
    def solve(self, alpha, u_inf = 1):
        B = np.zeros(len(self.RHS))

        # Normal vectors definition
        n_x, n_y = self.nx, self.ny

        # Freestream contribution
        B[:-1] = -(u_inf * np.cos(alpha) * n_x + u_inf * np.sin(alpha) * n_y)

        gammas = np.linalg.solve(self.RHS, B)
        return gammas
