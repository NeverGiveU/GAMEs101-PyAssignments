import numpy as np  


class Intersection(object):
    def __init__(self):
        self.happened = False 
        self.coords = np.array([0., 0., 0.], dtype=np.float32) # hit point
        self.normal = np.array([0., 0., 0.], dtype=np.float32) # hit normal
        self.distance = np.float("inf")
        self.obj = None # object
        self.mat = None # mterial