import math 
from tqdm import tqdm 
import numpy as np  
from utils import *


class HitPayload(object):
    def __init__(self):
        self.t_near = float("inf")
        self.index = 0 # integer
        self.uv = None # np.array --shape=(2,)
        self.hit_object = None # type=Object


class Renderer(object):
    def __init__(self):
        self.frame_buffer = None

    def render(self, scene):
        '''
        @param
            `scene` --type=Scene
        '''
        self.frame_buffer = np.zeros((scene.height, scene.width, 3), dtype=np.float64)
        half_fov= scene.fov/2 / 180*math.pi
        aspect_ratio = scene.width/scene.height # 成像的宽高比

        camera_pos = np.array([0., 0., 0.], dtype=np.float64)
        m = 0 
        # xy = np.zeros((scene.height, scene.width, 2))
        for iteration in tqdm(range(scene.height*scene.width)):
            i = iteration//scene.width
            j = iteration%scene.width
        # for i in range(scene.height):
        #     for j in range(scene.width):
            if True:
            # if i != 108 or j != 158:
            #     continue
            # else:
                x = ((j+.5)/(scene.width/2.)-1.)*math.tan(half_fov)*aspect_ratio
                y = ((-.5-i)/(scene.height/2.)+1.)*math.tan(half_fov)
                # xy[i, j, :] = np.array([x, y])
                # 从视点出发的 Ray 的方向
                direction = np.array([x, y, -1], dtype=np.float64)
                #                            ^ 设置成像位置为 -1.
                direction = direction/magnitude_a(direction) # 归一化
                # Ray Tracing
                shading = self.cast_ray(camera_pos, direction, scene, 0)
                self.frame_buffer[i][j] = shading
    
    def cast_ray(self, original, direction, scene, depth):
        if depth > scene.max_depth:
            return np.array([0., 0., 0.], dtype=np.float64)

        hit_color = scene.background_color # shape=(3,)
        payload = self.trace(original, direction, scene.objects)
        if payload is None:
            return hit_color
        # print("|||", payload.t_near, payload.uv)
        ## hit point
        hit_point = original + payload.t_near*direction
        normal, st = payload.hit_object.get_surface_properties(hit_point, direction, payload.index, payload.uv)
        ## shading
        material_type = payload.hit_object.material_type   
        f_inshadow = False   
        if material_type == "DIFFUSE_AND_GLOSSY":
            '''
            We use the Phong illumation model int the default case. 
            The phong model is composed of a diffuse and a specular reflection component.
            '''
            light_amt = 0.
            specular_color = 0.
            shadow_point_orig = (hit_point+scene.epsilon*normal) if a_dot_b(direction, normal) < 0 else hit_point-scene.epsilon*normal
            for light in scene.lights:
                direction_p2l = light.position - hit_point
                distance_p2l = a_dot_b(direction_p2l, direction_p2l) # 距离的平方
                direction_p2l = direction_p2l/distance_p2l**.5
                # print(shadow_point_orig, direction_p2l, distance_p2l)

                ldotn = max(0., a_dot_b(direction_p2l, normal))
                # print("<", direction_p2l, normal)
                # trace the shadow point
                # 以 hit_point 作为视点, 发射一条光线, 看是否会被其他物体遮挡住; 
                # 若否, 则环境光来自于光线直接照射
                # 若是, 则环境光为 0.
                # print(shadow_point_orig, direction_p2l)
                payload_shadow = self.trace(shadow_point_orig, direction_p2l, scene.objects)
                bool_inshadow = payload_shadow is not None and payload.t_near**2 < distance_p2l
                light_amt += 0. if bool_inshadow else light.intensity*ldotn
                # 镜面反射——这里计算高光的时候, 并没有考虑是否遮挡的问题
                direction_reflection = reflect(-direction_p2l, normal)
                specular_color += max(0., a_dot_b(direction_reflection, -direction))**payload.hit_object.specular_exponent * light.intensity
                # print(specular_color)
            # print(payload.hit_object.get_diffuse_color(st))
            hit_color = light_amt * payload.hit_object.get_diffuse_color(st) * payload.hit_object.Kd +\
                        specular_color * payload.hit_object.Ks 
            # print(hit_color)

        elif material_type == "REFLECTION_AND_REFRACTION":
            direction_reflection = normalize_a(reflect(direction, normal))
            direction_refraction = normalize_a(refract(direction, normal, payload.hit_object.ior)) 
            reflection_ray_orig = hit_point-normal*scene.epsilon if a_dot_b(direction_reflection, normal) < 0 else hit_point+normal*scene.epsilon
            refraction_ray_orig = hit_point-normal*scene.epsilon if a_dot_b(direction_refraction, normal) < 0 else hit_point+normal*scene.epsilon

            color_reflection = self.cast_ray(reflection_ray_orig, direction_reflection, scene, depth+1)
            color_refraction = self.cast_ray(refraction_ray_orig, direction_refraction, scene, depth+1)
            Kr = fresnel(direction, normal, payload.hit_object.ior)
            hit_color = color_reflection*Kr+color_refraction*(1-Kr)

        else: # "REFLECTION"
            Kr = fresnel(direction, normal, payload.hit_object.ior) 
            # 反射反向与
            direction_reflection = reflect(direction, normal)
            reflection_ray_orig = hit_point+normal*scene.epsilon if a_dot_b(direction_reflection, normal) < 0 else hit_point-normal*scene.epsilon
            
            hit_color = self.cast_ray(reflection_ray_orig, direction_reflection, scene, depth+1)
            hit_color = hit_color * Kr

        return hit_color 

    def trace(self, original, direction, objects):
        t_near = float("inf")
        payload = None
        for obj in objects:
            hit_or_not, t_near_tmp, index, uv = obj.intersect(original, direction)
            if hit_or_not == True and t_near_tmp is not None and t_near_tmp < t_near:
                if payload is None:
                    payload = HitPayload()
                payload.hit_object = obj
                payload.t_near = t_near_tmp
                t_near = t_near_tmp
                payload.index = index # 对象存储在 scene 中的下标
                payload.uv = uv       # 
        return payload





