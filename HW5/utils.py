import numpy as np  


MaterialType = {
    "DIFFUSE_AND_GLOSSY":0,
    "REFLECTION_AND_REFRACTION":1,
    "REFLECTION":2
}


def a_cross_b(a, b):
    return np.cross(a, b)

def a_dot_b(a, b):
    return (a*b).sum()

def magnitude_a(a):
    return np.sqrt(np.power(a, 2).sum())

def angle_between_a_and_b(a, b):
    angle = a_dot_b(a, b)
    angle = angle / (magnitude_a(a), magnitude_a(b))
    return math.acos(angle) # 返回弧度制

def a_proj2_b(a, b):
    bn = b/magnitude_a(b)
    return a_dot_b(a, bn) * bn 

def a_times_b(a, b):
    return a*b 

def normalize_a(a):
    return a/(magnitude_a(a)+1e-8)

def on_same_side(p1, p2, a, b):
    '''
    to see whether p1 and p2 are on the same sides of line `ab`
    '''
    cp1 = a_cross_b(b-a, p1-a)
    cp2 = a_cross_b(b-a, p2-a)
    c12 = a_dot_b(cp1, cp2)
    if c12 >= 0:
        return True 
    return False 

def get_triangle_normal(a, b, c):
    '''
    Input r the three vertices in clock-wise order of a triangle primitivity
    to calculate the normal of this triangle
    '''
    u = b-a
    v = c-a
    n = a_cross_b(u, v)
    return n / magnitude_a(n)

def inside_triangle(p, a, b, c):
    '''
    to see whether the point `p` is within a triangle of △abc
    '''
    # to see if it is within an infinite prism that the triangle outlines.
    within_tri_prisim = on_same_side(p, a, b, c) & on_same_side(p, b, a, c) & on_same_side(p, c, a, b)
    # if not, it will never be on the triangle
    if not within_tri_prisim:
        return False

    n = get_triangle_normal(a, b, c)
    p_proj2_n = a_proj2_b(p, n)
    mag = magnitude_a(p_proj2_n)
    if abs(magnitude_a) < 1e-6:
        return True 
    return False

def solve_quadratic(a, b, c):
    a = a+1e-8 if a >= 0 else a-1e-8
    delta = b*b - 4*a*c 
    if delta < 0:
        return False, 0., 0. 
    q = -(b+delta**0.5) if b > 0 else -(b-delta**0.5)
    x0 = q/(2*a)
    x1 = 2*c/q 
    if x0 > x1:
        x0, x1 = x1, x0
    return True, x0, x1

def reflect(dir_in, normal):
    '''
    @param
        `dir_in` --np.array --shape=(3,)
        `normal` --np.array --shape=(3,)
    '''
    return dir_in - 2*a_dot_b(dir_in, normal)*normal

def refract(dir_in, normal, ior):
    # print(">>in>>", dir_in, normal, ior)
    cos_i = a_dot_b(dir_in, normal)
    cos_i = min(max(-1., cos_i), 1.)
    eta_i = 1. 
    eta_t = ior 
    if cos_i < 0:
        cos_i = -cos_i
        normal_tmp = normal
    else:
        eta_i, eta_t = eta_t, eta_i
        normal_tmp = -normal # 由内向外射出，需要将法向量反向
    eta = eta_i/eta_t

    k = 1 - eta*eta*(1-cos_i*cos_i)
    # print(cos_i, eta, k)
    if k < 0:
        return np.array([0., 0., 0.], dtype=np.float64)
    else:
        return eta*dir_in + (eta*cos_i-k**.5)*normal_tmp

def fresnel(dir_in, normal, ior):
    ## 解菲涅尔方程
    '''
    @param:
        `dir_in` --np.array --shape=(3,) --help="入射光线方向"
    '''
    cos_i = a_dot_b(dir_in, normal)
    eta_i = 1. # 在真空中的折射率
    eta_t = ior# object 材料的折射率
    if cos_i > 0:
        eta_i, eta_t = eta_t, eta_i

    sin_t = eta_i / eta_t * (max(0., 1-cos_i**2))**.5 ## 根据反射定律计算折射角的正弦值
    if sin_t >= 1:
        return 1.
    cos_t = max(0., 1-sin_t**2)**.5
    cos_i = abs(cos_i)
    Rs = ((eta_t*cos_i)-(eta_i*cos_t)) / ((eta_t*cos_i)+(eta_i*cos_t))
    Rp = ((eta_i*cos_i)-(eta_t*cos_t)) / ((eta_i*cos_i)+(eta_t*cos_t))
    return (Rs*Rs+Rp*Rp)/2
