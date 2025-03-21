import numpy as np

class constant_source:
    def __init__(self, lambda_, p1, p2):
        # Convert inputs to numpy arrays
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        lambda_ = np.asarray(lambda_)
        
        # Ensure inputs are 2D arrays for multi-panel handling
        # This step allows us to use numpy's vectorization
        # so we don't loop over each panel twice when solving
        if p1.ndim == 1:
            p1 = p1.reshape(1, -1)
        if p2.ndim == 1:
            p2 = p2.reshape(1, -1)
        if lambda_.ndim == 0:
            lambda_ = lambda_.reshape(1)
        
        # Geometric vertex data (arrays)
        self.x1 = p1[:, 0]
        self.y1 = p1[:, 1]
        self.x2 = p2[:, 0]
        self.y2 = p2[:, 1]
        
        # Geometric panel data
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        self.length = np.hypot(dx, dy)
        self.alpha_panel = np.arctan2(dy, dx)

        self.lambda_ = lambda_
    
    def _transform_coordinates(self, x, y):
        # Translate and rotate coordinates for all panels
        x = np.asarray(x)[..., np.newaxis] 
        y = np.asarray(y)[..., np.newaxis]
        
        # Translation
        x_trans = x - self.x1
        y_trans = y - self.y1
        
        # Rotation
        alpha = -self.alpha_panel
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        x_p = x_trans * cos_alpha - y_trans * sin_alpha
        y_p = x_trans * sin_alpha + y_trans * cos_alpha
        
        return x_p, y_p

    def streamline(self, x, y):
        x_p, y_p = self._transform_coordinates(x, y)
        x1_panel, x2_panel = 0.0, self.length
        
        # Compute angles and radii
        theta1 = np.arctan2(y_p, x_p - x1_panel)
        theta2 = np.arctan2(y_p, x_p - x2_panel)
        r1_sq = (x_p - x1_panel)**2 + y_p**2
        r2_sq = (x_p - x2_panel)**2 + y_p**2

        psi = -self.lambda_ / (2 * np.pi) * (
            (x_p - x1_panel) * theta1 - (x_p - x2_panel) * theta2 + \
            y_p / 2 * np.log(r1_sq / r2_sq)
        )

        return np.sum(psi, axis=-1)

    def velocity(self, x, y):
        x_p, y_p = self._transform_coordinates(x, y)
        x1_panel, x2_panel = 0.0, self.length
        
        # Compute angles and radii
        theta1 = np.arctan2(y_p, x_p - x1_panel)
        theta2 = np.arctan2(y_p, x_p - x2_panel)
        r1_sq = (x_p - x1_panel)**2 + y_p**2
        r2_sq = (x_p - x2_panel)**2 + y_p**2

        u_p = self.lambda_ / (4 * np.pi) * np.log(r1_sq / r2_sq)
        v_p = self.lambda_ / (2 * np.pi) * (theta2 - theta1)

        cos_a = np.cos(-self.alpha_panel)
        sin_a = np.sin(-self.alpha_panel)
        
        u_global = u_p * cos_a + v_p * sin_a
        v_global = -u_p * sin_a + v_p * cos_a

        return u_global, v_global
