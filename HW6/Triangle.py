from OBJloader import OBJloader
from Material import Material
from utils import * 
from BBox3D import BBox3D
from BVH import *
from BVH import BoudingVolumeHierarchy as BVH
import numpy as np  
from Intersection import Intersection


class Object(object):
    def __init__(self):
        pass

    def get_intersection(self, ray):
        return Intersection()


class Triangle(Object):
    def __init__(self, v0, v1, v2, m):
        '''
        @param
            v0, v1, v2 --type=np.array --shape=(3,) 
            m --type=Material
        '''
        self.v0 = v0 
        self.v1 = v1 
        self.v2 = v2 
        self.m = m 

        self.e1 = v1-v0
        self.e2 = v2-v0

        self.normal = normalize_a(a_cross_b(self.e1, self.e2))

    def get_bbox3D(self):
        return bbox3D_union_vertex(BBox3D(self.v0, self.v1), self.v2)

    def get_intersection(self, ray):
        # print("Intersection to Triangle")
        intersection = Intersection()
        s0 = ray.original - self.v0 
        s1 = np.cross(ray.direction, self.e2)
        s2 = np.cross(s0, self.e1)

        s = np.array([
                a_dot_b(s2, self.e2),
                a_dot_b(s1, s0),
                a_dot_b(s2, ray.direction)
            ]) / a_dot_b(s1, self.e1)
        t_near = s[0]
        u = s[1]
        v = s[2]
        # print(t_near, u, v, self.m)
        if t_near >= 0. and u >= 0. and v >= 0. and (u+v) <= 1.:
            intersection.happened = True 
            intersection.coords = s 
            intersection.normal = self.normal
            intersection.mat = self.m 
            intersection.obj = self
            intersection.distance = t_near
        return intersection

    def get_surface_properties(self, hit_point, in_direction):
        return self.normal, np.zeros(3, dtype=np.float32), 0, np.zeros(2, dtype=np.float32)

    def get_diffuse_color(self, st):
        return np.array([.5, .5, .5])

    # def intersect(self, ray):
    #     pass



class MeshTriangle(Object):
    def __init__(self, file):
        print("Loading model ...")
        objloader = OBJloader()
        objloader.load_obj(file)
        
        print("\nFinishing loading. Info of the loaded object(s): ")
        print("    #mesh: ", len(objloader.loaded_meshes))
        print("#vertices: ", len(objloader.loaded_vertices))
        print(" #indices: ", len(objloader.loaded_indices))

        mesh = objloader.loaded_meshes[0]
        # calculate the most inner and the most outter vertix
        min_vertex = np.array([float("inf"), float("inf"), float("inf")], dtype=np.float32)
        max_vertex = np.array([float("inf"), float("inf"), float("inf")], dtype=np.float32) * -1

        self.triangles = []
        for i in range(0, len(mesh.vertices), 3):
            positions = []
            for j in range(3):
                vertex = mesh.vertices[i+j]
                position = vertex.position * 60. 
                positions.append(position)

                # 更新最里与最外的坐标
                min_vertex = np.minimum(min_vertex, position)
                max_vertex = np.maximum(max_vertex, position)
            # 给这个三角形增加材质
            tri_material = Material(0, color=np.array([.5, .5, .5], dtype=np.float32), emission=np.array([0., 0., 0.], dtype=np.float32))
            # tri_material = Material(1, color=np.array([.5, .5, .5], dtype=np.float32), emission=np.array([0., 0., 0.], dtype=np.float32))
            # tri_material = Material(2, color=np.array([.5, .5, .5], dtype=np.float32), emission=np.array([0., 0., 0.], dtype=np.float32))
            tri_material.Kd = .6 
            tri_material.Ks = .0 
            tri_material.specular_exponent = 0
            self.triangles.append(Triangle(*positions, tri_material))

        ## 包围盒
        self.bbox = BBox3D(min_vertex, max_vertex)
        self.bvh = BVH(self.triangles)

    def get_bbox3D(self):
        return self.bbox

    def get_intersection(self, ray):
        # print("Intersection to MeshTriangle")
        return self.bvh.intersect(ray)

    def get_surface_properties(self, hit_point, in_direction):
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0, np.zeros(2, dtype=np.float32)


    # def intersect(self, ray):
    #     t_near = float("inf")
    #     uv = None
    #     index = -1
    #     i = 0
    #     hit_happened = False
    #     for triangle in self.triangles:
    #         happened, t_near_tmp, uv_tmp = ray_triangle_intersect(triangle.v0, triangle.v1, triangle.v2,
    #                                                       ray.original, ray.direction)
    #         if happened and t_near > t_near_tmp:
    #             index = i 
    #             t_near = t_near_tmp
    #             uv = uv_tmp
    #             hit_happened = happened
    #         i += 1
    #     return hit_happened, index, uv 

    def get_diffuse_color(self, st):
        scale = 5.
        pattern = (st[0]*scale%1. > 0.5)^(st[1]*scale%1. > 0.5)
        if pattern:
            return np.array([.937, .937, .231])
        else:
            return np.array([.815, .235, .031])
        


def ray_triangle_intersect(v0, v1, v2, orig, dir):
    '''
    @description: 判断光线与三角形是否有交点
    @param:
        all parameters --np.array --shape=(3,)
    '''
    e1 = v1-v0
    e2 = v2-v0
    n = a_cross_b(e1, e2)
    n = n/magnitude_a(n)

    ## 计算交点
    '''
    ray: r = orig + t*dir
    plane: (orig + t*dir - v0).dot(n) = 0.

    ->

    t = (v0-orig).dot(n) / (dir.dot(n))
    '''
    # t = a_dot_b(v0-orig, n) / a_dot_b(dir, n)
    # if t <= 0:
    #     return False, None, None 
    # p = orig + t*dir # the hit point

    ## 计算重心坐标, 顺带可以帮助判断交点是否在 △ 内
    # 以下代码借鉴于: https://github.com/feichang2/games101hw
    # ⚠️⚠️使用 tutorial 13, P29 中的 Moller Trumbore Algorithm 算法
    s = orig-v0
    s1 = np.cross(dir,e2)
    s2 = np.cross(s,e1)
    s1e1 = np.dot(s1,e1)
    if s1e1 == 0.:
        s1e1 += 1e-8
    t = np.dot(s2,e2)/s1e1
    u = np.dot(s1,s)/s1e1
    v = np.dot(s2,dir)/s1e1

    if t > 0 and 0 <= u <= 1 and 0 <= v <= 1 and 0 <= u+v <= 1: # 判断解的有效性
        return True, t, np.array([u, v])
    return False, None, None 




