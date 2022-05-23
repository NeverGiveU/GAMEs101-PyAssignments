import numpy as np  


class Ray(object):
    def __init__(self, original, direction, t=0.):
        self.original = original
        self.direction = direction
        self.t = t 
        self.t_min = 0. 
        self.t_max = float("inf")

        self.inv_direction = np.array([1./(direction[0]+1e-6), 1./(direction[1]+1e-6), 1./(direction[2]+1e-6)])

    def emission(self, t):
        return self.original + t * self.direction
