import numpy as np  


MaterialType = ["DIFFUSE_AND_GLOSSY", "REFLECTION_AND_REFRACTION", "REFLECTION"]

class Material(object):
    def __init__(self, material_type=0, color=np.zeros(3), emission=np.zeros(3)):
        self.mtype = MaterialType[material_type]
        self.mcolor = color 
        self.memission = emission

        self.ior = 1.3#0.
        self.Kd = 0.
        self.Ks = 0.
        self.specular_exponent = 0.