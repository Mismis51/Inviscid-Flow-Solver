import numpy as np

class linear_vortex:
    def __init__(self, gamma_a, gamma_b, p1, p2):
        # Convert inputs to numpy arrays
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        gamma_a = np.asarray(gamma_a)
        gamma_b = np.asarray(gamma_b)
        
        # Ensure inputs are 2D arrays for multi-panel handling
        # This step allows us to use numpy's vectorization
        # so we don't loop over each panel twice when solving
        if p1.ndim == 1:
            p1 = p1.reshape(1, -1)
        if p2.ndim == 1:
            p2 = p2.reshape(1, -1)
        if gamma_a.ndim == 0:
            gamma_a = gamma_a.reshape(1)
        if gamma_b.ndim == 0:
            gamma_b = gamma_b.reshape(1)
        
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
        
        # Vortex strengths handling
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        
        self.gamma0 = self.gamma_a
        self.gamma1 = (self.gamma_b - self.gamma_a) / self.length
    
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
        
        # Linear term components
        psi_linear = self.gamma1 / (4 * np.pi) * (
            (x_p**2 - x1_panel**2 - y_p**2) / 2 * np.log(r1_sq)
            - (x_p**2 - x2_panel**2 - y_p**2) / 2 * np.log(r2_sq)
            + 2 * x_p * y_p * (theta2 - theta1)
            - x_p * (x2_panel - x1_panel)
        )
        
        # Constant term components
        psi_constant = self.gamma0 / (4 * np.pi) * (
            (x_p - x1_panel) * np.log(r1_sq)
            - (x_p - x2_panel) * np.log(r2_sq)
            + 2 * y_p * (theta2 - theta1)
        )
        
        # Sum contributions from all panels
        # No need to rotate back, since we are computing a scalar
        return np.sum(psi_constant + psi_linear, axis=-1)
    
    def velocity(self, x, y):
        x_p, y_p = self._transform_coordinates(x, y)
        x1_panel, x2_panel = 0.0, self.length
        
        # Compute angles and radii
        theta1 = np.arctan2(y_p, x_p - x1_panel)
        theta2 = np.arctan2(y_p, x_p - x2_panel)
        r1_sq = (x_p - x1_panel)**2 + y_p**2
        r2_sq = (x_p - x2_panel)**2 + y_p**2
        
        # Linear velocity components
        u_lin = -self.gamma1 / (4 * np.pi) * (
            y_p * np.log(r1_sq / r2_sq) - 2 * x_p * (theta2 - theta1)
        )
        v_lin = -self.gamma1 / (2 * np.pi) * (
            .5 * x_p * np.log(r1_sq / r2_sq) - self.length + y_p * (theta2 - theta1)
        )

        # Constant velocity components
        u_const = self.gamma0 / (2 * np.pi) * (theta2 - theta1)
        v_const = self.gamma0 / (4 * np.pi) * np.log(r2_sq / r1_sq)
        
        # Combine and rotate to global coordinates
        u_p = u_lin + u_const
        v_p = v_lin + v_const
        
        cos_a = np.cos(-self.alpha_panel)
        sin_a = np.sin(-self.alpha_panel)
        
        u_global = u_p * cos_a + v_p * sin_a
        v_global = -u_p * sin_a + v_p * cos_a
        
        return u_global, v_global
