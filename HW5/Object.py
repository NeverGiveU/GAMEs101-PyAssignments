from utils import *


class Object(object):
    def __init__(self):
        self.ior = 1.3
        self.Kd = .8
        self.Ks = .2
        self.color_diffuse = np.array([.2, .2, .2]) # --type=np.array --shape=(3,)
        self.specular_exponent = 25.

        self.material_type = "DIFFUSE_AND_GLOSSY"

    def get_diffuse_color(self, st):
        return self.color_diffuse


class Sphere(Object):
    def __init__(self, center, radius):
        super(Sphere, self).__init__()
        '''
        @param
            `center` --np.array --shape=(3,1)
            `radius` --type=float
        '''
        self.center = center
        self.r = radius
        self.r_square = self.r*self.r 

    def intersect(self, orig, dir):
        '''
        @param
            `orig` --np.array --shape=(3,)
            `dir` --np.array --shape=(3,)
        '''
        L = orig - self.center # 从视点指向球心
        '''
        x = oirg+t*dir
        (x-ctr)^2 = r^2
        
        ->

        (orig+t*dir-ctr)^2 = [(orig-ctr)+t*dir]^2 = (L+t*dir)^2 = r^2

        ->

        dir^2*t^2 + 2L.dot(dir)*t + L^2-r^2 = 0.
        ^^^^^a      ^^^^^^^^^^^b    ^^^^^^^c
        '''
        a = a_dot_b(dir, dir)  
        b = 2*a_dot_b(dir, L)  
        c = a_dot_b(L, L) - self.r_square     # 视点到球心的距离
        f, t0, t1 = solve_quadratic(a, b, c)  # 解二次方程
        if not f:
            return False, None, None, None
        if t0 < 0:
            t0 = t1
        if t0 < 0:
            return False, None, None, None 
        return True, t0, 0, np.array([0., 0.], dtype=np.float64)

    def get_surface_properties(self, hit_point, direction, index, uv):
        normal = hit_point-self.center
        normal = normal/magnitude_a(normal)
        return normal, np.array([0., 0.], dtype=np.float64)


class MeshTriangle(Object):
    def __init__(self, vertices, indices, n_triangles, st):
        super(MeshTriangle, self).__init__()
        '''
        @param
            `vertices` --np.array --shape=(N,3)
            `indices` --list<int> --len=n_triangles*3
            `n_triangles` --int
            `st` --np.array --shape=(N,2) --description="uv-coordinates"
        '''
        self.vertices = vertices
        self.indices = indices
        self.n_triangles = n_triangles
        self.st_coordinates = st 

    def intersect(self, orig, dir):
        t_near = float("inf")
        index = 0
        intersect_or_not = False 
        # print(">>", orig, dir)
        uv = None

        for i in range(self.n_triangles):
            i0, i1, i2 = self.indices[i*3:i*3+3]
            v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
            
            intersect_or_not_tmp, t_near_tmp, uv_tmp = ray_triangle_intersect(v0, v1, v2, orig, dir)
            # print("<<<<", t_near_tmp, t_near, uv)
            if intersect_or_not_tmp and t_near_tmp < t_near:
                index = i  
                t_near = t_near_tmp
                uv = uv_tmp
                intersect_or_not = True
        # print(">>", t_near, uv, index)

        return intersect_or_not, t_near, index, uv

    def get_surface_properties(self, hit_point, direction, index, uv):
        i0, i1, i2 = self.indices[index*3:index*3+3]
        v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
        e01 = v1-v0
        e12 = v2-v1
        e01 = e01/(magnitude_a(e01)+1e-8)
        e12 = e12/(magnitude_a(e12)+1e-8)

        normal = a_cross_b(e01, e12)
        normal = normal/(magnitude_a(normal)+1e-8)

        ## 纹理坐标
        st0, st1, st2 = self.st_coordinates[i0], self.st_coordinates[i1], self.st_coordinates[i2]
        u, v = uv 
        st = st0*(1-u-v) + st1*u + st2*v 

        return normal, st 

    def get_diffuse_color(self, st):
        '''
        @param:
            `st` --np.array --shape=(2,)
        '''
        # print(st)
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


class Light(object):
    def __init__(self, position, intensity):
        '''
        @param
            `position` --np.array --shape=(3,)
            `intensity` --np.array --shape=(3,)
        '''
        self.position = position
        self.intensity = intensity