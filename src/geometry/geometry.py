import numpy as np
from ._interpolate import _interpolate
from ._transformations import _normalize_and_center, _rotate_around_ahalf
from ._naca import _naca4

class geometry:
    def __init__(self, nb_vertex=256, angle=0):
        self.nb_vertex = nb_vertex
        self.angle = angle

    def _sign(self):
        # We will be checking the sign convention at the half point
        # If 0 width, raise an eror
        idx = self.nb_vertex // 4
        # This means we are going anti clockwise
        if self.vertex[idx // 4, 1] > self.vertex[-idx // 4, 1]:
            return 1
        # We are going clockwise
        elif self.vertex[idx // 4, 1] < self.vertex[-idx // 4, 1]:
            return -1
        else:
            raise ValueError('Geometry must have a non 0 width at half point')

    def _compute_parameters(self):
        diff = np.roll(self.vertex, -1, axis=0) - self.vertex
        self.ds = np.linalg.norm(diff, axis=1)
        self.center = self.vertex + diff * .5
        
        self.normal = self._sign() * np.column_stack((diff[:, 1], -diff[:, 0]))
        self.normal /= np.linalg.norm(self.normal, axis=1)[:, np.newaxis]
    
    def _initialize(self):
        self.vertex = _interpolate(self.vertex, self.nb_vertex)
        self.vertex = _normalize_and_center(self.vertex)
        self._compute_parameters()

    def load_txt(self, data):
        self.vertex = np.loadtxt(data)
        self._initialize()

    def load_naca(self, m, p, t):
        # It will be interpolated to desired points afterwards
        x = np.linspace(0, 1, self.nb_vertex // 2)
        self.vertex = _naca4(x, m, p, t)
        self._initialize()

    def set_angle_deg(self, angle_deg):
        self.angle = angle_deg * np.pi / 180

    def set_angle_rad(self, angle_rad):
        self.angle = angle_rad

    def get_rotated_vertex(self):
        return _rotate_around_ahalf(self.vertex, self.angle)
    
    def update_nbpoints(self, new_nb):
        self.nb_vertex = new_nb
        self._initialize()
