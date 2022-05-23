import numpy as np  
from BVH import BoudingVolumeHierarchy as BVH


class Scene(object):
    def __init__(self, width=1280, height=960):
        self.width = width
        self.height= height
        self.fov = 90 
        self.background_color = np.array([0.235294, 0.67451, 0.843137])
        self.max_depth = 5 
        self.epsilon=1e-5

        self.objects = []
        self.lights = [] 
        self.bvh = None

    def add_object(self, object):
        '''
        @param
            `object` --type=Object
        '''
        self.objects.append(object)

    def add_light(self, light):
        '''
        @param
            `light` --type=Light
        '''
        self.lights.append(light)

    def buildBVH(self):
        print("\nConstructing BVH ...")
        self.bvh = BVH(self.objects, 1, 0)