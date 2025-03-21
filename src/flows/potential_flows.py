import numpy as np

class freestream:
    def __init__(self, u_inf):
        self.u_inf = u_inf

    def streamline(self, x, y):
        return self.u_inf * y
    
    def velocity(self, x, y):
        u = self.u_inf * np.ones_like(x)
        v = np.zeros_like(y)
        return u, v

class source:
    def __init__(self, _lambda, x, y):
        self._lambda = _lambda
        self.x = x
        self.y = y

    def streamline(self, x, y):
        theta = np.arctan2(y - self.y, x - self.x) 
        return self._lambda / (2 * np.pi) * (theta + 2 * np.pi * (theta < 0))

class vortex:
    def __init__(self, gamma, x, y):
        # Ensure gamma, x, and y are numpy arrays
        self.gamma = np.asarray(gamma)  # shape: (M,)
        self.x = np.asarray(x)          # shape: (M,)
        self.y = np.asarray(y)          # shape: (M,)

    def streamline(self, X, Y):
        """
        Compute the streamfunction at points (X, Y) due to all vortices.
        X and Y can be scalars or numpy arrays of any shape.
        The output will have the same shape as X and Y.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        # Compute the squared distance from each query point to each vortex.
        # The resulting shape will be (..., M) where M is the number of vortices.
        r2 = (X[..., None] - self.x)**2 + (Y[..., None] - self.y)**2
        # Compute the contribution from each vortex.
        # Note: 1/2 factor is preserved from your original code.
        psi = self.gamma / (2 * np.pi) * 0.5 * np.log(r2)
        # Sum the contributions over all vortices.
        return np.sum(psi, axis=-1)

    def velocity(self, X, Y):
        """
        Compute the velocity (u, v) at points (X, Y) due to all vortices.
        Returns two arrays (u, v) with the same shape as X and Y.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        dx = X[..., None] - self.x  # shape: (..., M)
        dy = Y[..., None] - self.y  # shape: (..., M)
        r2 = dx**2 + dy**2
        # Calculate the velocity contributions from each vortex.
        u = self.gamma / (2 * np.pi) * dy / r2
        v = -self.gamma / (2 * np.pi) * dx / r2
        # Sum the contributions from all vortices.
        return np.sum(u, axis=-1), np.sum(v, axis=-1)

class doublet:
    def __init__(self, kappa, x, y):
        self.kappa = kappa
        self.x = x
        self.y = y
    
    def streamline(self, x, y):
        r2 = (x - self.x)**2 + (y - self.y)**2
        theta = np.arctan2(y - self.y, x - self.x)
        return -self.kappa / (2 * np.pi) * (np.sin(theta)) / np.sqrt(r2)