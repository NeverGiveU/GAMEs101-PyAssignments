import time 
from BVH import *
from BBox3D import *
from Intersection import Intersection
# from Triangle import *


SplitMethod = ["NAIVE", "SAH"]

class BVHNode(object):
    def __init__(self):
        self.bbox = None # --type=BBox3D
        self.left = None # --description=left_child
        self.right = None# --description=right_child
        self.object= None# --type=Object

        self.split_axis = 0
        self.first_primitity_offset = 0.
        self.n_primitives = 0


class BoudingVolumeHierarchy(object):
    def __init__(self, pointers, max_primitity_in_node=1, split_method=0):
        self.split_method = SplitMethod[split_method]
        self.max_primitity_in_node = min(255, max_primitity_in_node)
        self.primitives = pointers

        assert len(self.primitives) > 0
        stime = time.time()
        self.root = self.recursive_building(self.primitives)
        etime = time.time()
        ## 统计用时
        dtime = etime-stime
        h = int(dtime)//3600
        m = int(dtime)//60 - h*60
        s = dtime - h*3600 - m*60
        print("\nBVH Generation completed. Time taken: %02d:%02d:%.4f."%(h, m, s))

    def recursive_building(self, objects):
        if len(objects) == 0:
            return None 

        node = BVHNode()
        ## bbox
        bbox = BBox3D()
        for i in range(len(objects)):
            obj = objects[i]
            bbox = bbox3D_union_bbox3D(bbox, obj.get_bbox3D())
            # print(bbox.min_vertex, bbox.max_vertex)
        if len(objects) == 1:
            # single object, save as a leaf node and stop
            node.bbox = bbox 
            node.object = objects[0]

        elif len(objects) == 2:
            node.left = self.recursive_building(objects[:1])
            node.right= self.recursive_building(objects[1:])
            node.bbox = bbox3D_union_bbox3D(node.left.bbox, node.right.bbox)

        else:
            centroid_bbox = BBox3D()
            for i in range(len(objects)):
                obj = objects[i]
                bbox = bbox3D_union_vertex(bbox, obj.get_bbox3D().centroid()) 
            dim = centroid_bbox.max_extent() # 较长轴
            if dim == 0:
                objects = sorted(objects, key=lambda obj:obj.get_bbox3D().centroid()[0])
            elif dim == 1:
                objects = sorted(objects, key=lambda obj:obj.get_bbox3D().centroid()[1])
            else: # dim == 2
                objects = sorted(objects, key=lambda obj:obj.get_bbox3D().centroid()[2])

            m = len(objects)//2
            node.left = self.recursive_building(objects[:m])
            node.right= self.recursive_building(objects[m:])
            node.bbox = bbox3D_union_bbox3D(node.left.bbox, node.right.bbox)

        return node 

    def intersect(self, ray):
        '''
        @param
            `ray` --type=Ray
        '''
        intersection = Intersection()
        if self.root is None:
            return intersection
        intersection = self.get_intersection(self.root, ray)
        return intersection

    def get_intersection(self, root, ray):
        dir_is_neg = []
        for i in range(3):
            if ray.direction[i] >= 0:
                dir_is_neg.append(1)
            else:
                dir_is_neg.append(0)
        dir_is_neg = np.array(dir_is_neg, dtype=np.uint8)

        intersection = Intersection()
        # 与当前节点的包围盒检查相交情况
        if root.bbox.intersect(ray, ray.inv_direction, dir_is_neg) is False:
            return intersection 

        if root.left is None and root.right is None:
            # 叶子节点
            intersection = root.object.get_intersection(ray)
            return intersection

        hit1 = self.get_intersection(root.left, ray)
        hit2 = self.get_intersection(root.right, ray)

        return hit1 if hit1.distance < hit2.distance else hit2 
