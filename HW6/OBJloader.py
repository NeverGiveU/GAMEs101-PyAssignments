import math  
import numpy as np  
import os 
from tqdm import tqdm  
import re 


class Vertex(object):
    def __init__(self, position=None, normal=None, texture_coordinate=None):
        '''
        @param 
            position --np.array --shape=(3,)
            normal   --np.array --shape=(3,)
            texture_coordinate --np.array --shape=(2,)
        '''
        self.position = position
        self.normal = normal
        self.texture_coordinate = texture_coordinate


class Material(object):
    def __init__(self):
        self.name = ""
        self.Ns = 0.   # float
        self.Ni = 0.   # float
        self.d = 0.    # float
        self.illum = 0 # illumination, int

        self.K_Ambient = np.zeros(3)
        self.K_Diffuse = np.zeros(3)
        self.K_Specular = np.zeros(3)

        self.map_Ambient_path = ""           # Ambient Texture Map
        self.map_Diffuse_path = ""           # Diffuse Texture Map
        self.map_Specular_path = ""          # Specular Texture Map
        self.map_SpecularHighlight_path = "" # Specular Hightlight Map
        self.map_Alpha_path = ""             # Alpha Texture Map  
        self.map_bumping_path = ""           # Bump Map 


class Mesh(object):
    '''
    A simple Mesh Object that holds a name, a vertex list, and an index list.
    '''
    def __init__(self):
        '''
        @param
            vertices --list<Vertex> --len=N
            indices  --np.array --dtype=np.uint8 --shape=(N,)
        '''
        self.vertices = []
        self.indices = []
        self.name = ""
        self.material = ""


## some vector operations
'''
@param 
    a --np.array --shape=(3,)
    b --np.array --shape=(3,)
'''
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


## 字符串解析函数
def first_token(s):
    if len(s) > 0:
        tokens = re.split(r"[ ]+", s)
        for token in tokens:
            if len(token) > 0:
                return token
    return ""

def last_token(s):
    if len(s) > 0:
        tokens = re.split(r"[ ]+", s)
        for token in tokens[::-1]:
            if len(token) > 0:
                return token
    return ""


class OBJloader(object):
    def __init__(self):
        self.loaded_meshes = []    # list<Mesh>
        self.loaded_vertices = []  # list<Vertex>
        self.loaded_indices = []   # np.array
        self.loaded_materials = [] # list<Material>

    def load_obj(self, obj_path):
        if os.path.basename(obj_path).split(".")[-1] != "obj":
            return False 
        self.loaded_meshes = []    
        self.loaded_vertices = []  
        self.loaded_indices = []   

        positions = []
        tex_coords = []
        normals = []

        vertices = []
        indices = []

        meshnames = []
        mesh = Mesh()

        listening = False 
        meshname = "";

        ## begin to read
        log_freq = 1000
        iteration = log_freq

        fh = open(obj_path)
        i = 0
        cache = {}
        lines = fh.readlines()
        fh.close()
        count = 0
        for line in tqdm(lines):
            line = line[:-1]
            if len(line) <= 0:
                continue
            token = first_token(line)
            cache[token] = True 

            if token == "o" or token == "g" or line[0] == "g":
                if not listening:
                    listening = True 
                    if token == "o" or token == "g":
                        meshname = last_token(line) 
                    else:
                        meshname = "unnamed"
                else:
                    if len(indices) > 0 and len(vertices) > 0:
                        mesh = Mesh()
                        mesh_tmp.name = meshname
                        mesh_tmp.vertices = vertices
                        mesh_tmp.indices = indices
                        self.loaded_meshes.append(mesh)

                        vertices = []
                        indices = []
                        meshname = last_token(line)
                    else:
                        if token == "o" or token == "g":
                            meshname = last_token(line)
                            # print(">>", meshname)
                        else:
                            meshname = "unnamed"
            elif token == "v":
                x, y, z = line.split(" ")[1:4]
                vertex = np.array([float(x), float(y), float(z)])
                positions.append(vertex)
            elif token == "vt":
                u, v = line.split(" ")[1:3]
                tex_coord = np.array([float(u), float(v)])
                tex_coords.append(tex_coord)
            elif token == "vn":
                xn, yn, zn = line.split(" ")[1:4]
                normal = np.array([float(xn), float(yn), float(zn)])
                normals.append(normal)
            elif token == "f":
                # print("#######################################################")
                # print("#positions:", len(positions))
                # print("#tex_coords:", len(tex_coords))
                # print("#normals:", len(normals))
                # we have got vertices w/i their attributes.
                # now we need to link them to form triangles
                ## add vertices
                vertices_tmp = self.gen_vertices_from_raw_obj(positions, tex_coords, normals, line)
                vertices.extend(vertices_tmp)
                self.loaded_vertices.extend(vertices_tmp)
                
                ## add inidices
                indices_tmp = self.vertices_triangluation(vertices_tmp)  # 长度一定是 3 的倍数
                for index in indices_tmp:
                    # vertices 和 self.loaded_vertices 的区别在于: 前者是每个 mesh 特有的, 后者是全部 meshes 的并集
                    # 一个文件中可能存在多个 mesh, i.e., 对应多个对象 
                    ind_number = len(vertices)-len(vertices_tmp) + index
                    '''
                    简单介绍这种记法:
                        假如 vertices 中原先就有 {v1, v2, ..., vm}
                        现在有 4 个新的节点放在 indices_tmp 中        # from `self.gen_vertices_from_raw_obj`
                        扩展到 vertices 后就有: {v1, v2, ..., vm; vm1, vm2, vm3, vm4}

                        经过经过三角形化处理后, 得到一些三角形的顶点 indices_tmp = [0, 1, 3, 0, 1, 2]
                        于是对应的 ind_number 就是取值: m+0, m+1, m+3, m+0, m+1, m+2
                        则恰好是对应扩展后的 vertices 中的新的节点
                    '''
                    indices.append(ind_number)

                    ind_number = len(self.loaded_vertices)-len(vertices_tmp) + index
                    self.loaded_indices.append(ind_number)

            if token == "usemtl":
                meshnames.append(last_token(line))
                if len(indices) > 0 and len(vertices) > 0:
                    mesh_tmp = Mesh()
                    mesh_tmp.name = meshname
                    mesh_tmp.vertices = vertices
                    mesh_tmp.indices = indices
                    
                    a = 2
                    # while True:
                    mesh_tmp.name = meshname + "_{}".format(a)
                    for mesh in self.loaded_meshes:
                        if mesh.name == mesh_tmp.name:
                            continue
                        # break
                    self.loaded_meshes.append(mesh_tmp)
                    vertices = []
                    indices = []

            if token == "mtllib":
                mat_path = obj_path.replace(os.path.basename(obj_path), last_token(line))
                print("Loading material from `{}` ...".format(mat_path))

                f = self.load_material(mat_path)
                if f:
                    print("Successfully loading the material.")
        ## outside the travesal

        if len(indices) > 0 and len(vertices) > 0:
            mesh_tmp = Mesh()
            mesh_tmp.name = meshname
            mesh_tmp.vertices = vertices
            mesh_tmp.indices = indices
            self.loaded_meshes.append(mesh_tmp)

        ## Assign metarial to each mesh
        # print(len(self.loaded_vertices))
        # print(len(self.loaded_indices))
        # print(len(self.loaded_materials))
        # print(len(self.loaded_meshes))
        for i in range(len(meshnames)):
            meshname = meshnames[i]
            for j in range(len(self.loaded_meshes)):
                if meshname == self.loaded_meshes[j].name:
                    loaded_meshes[i].material = self.loaded_materials[j]
                    break

        if len(self.loaded_meshes) <= 0 and\
           len(self.loaded_vertices) <= 0 and\
           len(self.loaded_indices) <= 0:
            return False
        else:
            return True 
            
    def load_material(self, mat_path):
        assert mat_path.endswith("mtl"), "Material file must end with `.mtl`."
        fh = open(mat_path, "r")
        lines = fh.readlines()
        fh.close()

        listening = False 
        material_tmp = Material()
        for line in tqdm(lines):
            line = line[:-1]
            token = first_token(line)
            if token == "newmtl":
                if not listening:
                    listening = True
                    if len(line) > 7:
                        material_tmp.name = last_token(line)
                    else:
                        material_tmp.name = "none"
                else:
                    self.loaded_materials.append(material_tmp)
                    material_tmp = Material()
                    if len(line) > 7:
                        material_tmp.name = last_token(line)
                    else:
                        material_tmp.name = "none"

            ## Ambient color
            if token == "Ka":
                _, x, y, z = line.split(" ")
                material_tmp.K_Ambient = np.array([float(x), float(y), float(z)])
            ## Diffuse color
            if token == "Kd":
                _, x, y, z = line.split(" ")
                material_tmp.K_Diffuse = np.array([float(x), float(y), float(z)])
            ## Specular color
            if token == "Ks":
                _, x, y, z = line.split(" ")
                material_tmp.K_Specular = np.array([float(x), float(y), float(z)])
            ## Specular Exponent
            if token == "Ns":
                material_tmp.Ns = float(last_token(line))
            ## Optical Density
            if token == "Ni":
                material_tmp.Ni = float(last_token(line))
            ## Dissolve
            if token == "d":
                material_tmp.d = float(last_token(line))
            ## Illumination
            if token == "illum":
                material_tmp.illum = float(last_token(line))
            ## Ambient texture map
            if token == "map_Ka":
                material_tmp.map_Ambient_path = last_token(line)
            ## Diffuse texture map
            if token == "map_Kd":
                material_tmp.map_Diffuse_path = last_token(line)
            ## Specular texture map
            if token == "map_Ks":
                material_tmp.map_Specular_path = last_token(line)
            ## Specular highlight map
            if token == "map_Ns":
                material_tmp.map_SpecularHighlight_path = last_token(line)
            ## Alpha texture map
            if token == "map_d":
                material_tmp.map_Alpha_path = last_token(lines)
            ## Bump map
            if token == "map_Bump" or token == "map_bump" or token == "bump":
                material_tmp.map_bumping_path = last_token(line)

        self.loaded_materials.append(material_tmp)
        return len(self.loaded_materials) > 0

    def gen_vertices_from_raw_obj(self, ipositions, itex_coords, inormals, s):
        '''
        @param
            ipositions --list<np.array/shape=(3,)>
            itex_coords --list<np.array/shape=(2,)>
            inormals --list<np.array/shape=(3,)>
            s --str
        @return
            vertices --list<Vertex>

        '''
        vertices = []
        groups = s.split(" ")[1:]
        wo_normal = False
        for group in groups:
            ids = group.split("/")
            if len(ids) == 1:
                vtype = 1
            elif len(ids) == 2:
                vtype = 2
            elif len(ids) == 3:
                if ids[1] != "":
                    vtype = 4
                else:
                    vtype = 3
            # print(vtype)
            
            if vtype == 1: # P
                vertex = Vertex(position=self.get_element(ipositions, int(ids[0])))
                vertices.append(vertex)
                wo_normal = True
            elif vtype == 2: # P/T
                vertex = Vertex(position=self.get_element(ipositions, int(ids[0])),
                                texture_coordinate=self.get_element(itex_coords, int(ids[1])))
                vertices.append(vertex)
                wo_normal = True
            elif vtype == 3: # P//N
                vertex = Vertex(position=self.get_element(ipositions, int(ids[0])),
                                normal=self.get_element(inormals, int(ids[2])),
                                texture_coordinate=np.zeros(2, dtype=np.float32))
                vertices.append(vertex)
            elif vtype == 4: # P/T/N
                vertex = Vertex(position=self.get_element(ipositions, int(ids[0])),
                                normal=self.get_element(inormals, int(ids[2])),
                                texture_coordinate=self.get_element(itex_coords, int(ids[1])))
                vertices.append(vertex)
        
        if wo_normal:
            a = vertices[0].position - vertices[1].position
            b = vertices[2].position - vertices[1].position

            normal = a_cross_b(a, b)
            for vertex in vertices:
                vertex.normal = normal 

        return vertices

    def get_element(self, sequence, _id):
        if _id < 0:
            _id = len(sequence)+_id 
        else:
            _id -= 1
        return sequence[_id]

    def vertices_triangluation(self, vertices): 
        '''
        @param
            vertices --type=list<Vertex>
        '''
        n = len(vertices)
        if n < 3:
            return None
    
        if n == 3:
            indices = [0, 1, 2]
            return indices

        tvertices = vertices
        indices = []

        while True:
            for i in range(len(tvertices)):
                '''
                以 4 个节点为例
                   v0, v1, v2, v3
                    ^   ^       ^
                   *c  *n      *p
                            ^
                           tmp
                '''
                prev_vertex = vertices[i-1]
                curr_vertex = vertices[i]
                next_vertex = vertices[(i+1)%n]

                if len(tvertices) == 3:
                    for j in range(len(tvertices)):
                        if eq_npfloat32(vertices[j].position, curr_vertex.position):
                            indices.append(j)
                        if eq_npfloat32(vertices[j].position, prev_vertex.position):
                            indices.append(j)
                        if eq_npfloat32(vertices[j].position, next_vertex.position):
                            indices.append(j)
                    tvertices = []#[None]*len(vertices)
                    break

                temp_vertex = np.zeros(3)
                if len(tvertices) == 4:
                    for j in range(len(tvertices)):
                        if eq_npfloat32(vertices[j].position, curr_vertex.position):
                            indices.append(j)
                        if eq_npfloat32(vertices[j].position, prev_vertex.position):
                            indices.append(j)
                        if eq_npfloat32(vertices[j].position, next_vertex.position):
                            indices.append(j)
                        # 先取到: 0, 1, 3

                    for j in range(len(tvertices)):# 找到剩下一个为用到的节点
                        if eq_npfloat32(tvertices[j].position, curr_vertex.position) == False and \
                           eq_npfloat32(tvertices[j].position, prev_vertex.position) == False and \
                           eq_npfloat32(tvertices[j].position, next_vertex.position) == False:
                            temp_vertex = tvertices[j].position
                            break

                    for j in range(len(vertices)):
                        if eq_npfloat32(vertices[j].position, prev_vertex.position):
                            indices.append(j)
                        if eq_npfloat32(vertices[j].position, next_vertex.position):
                            indices.append(j)
                        if eq_npfloat32(vertices[j].position, temp_vertex):
                            indices.append(j)

                    tvertices = []#[None]*len(vertices)
                    # print(">> TO break out..")
                    break

                ## 如果超过了 4 个结点
                angle = angle_between_a_and_b(prev_vertex.position-curr_vertex.position,
                                              next_vertex.position-curr_vertex.position)*(180.0/math.pi)
                if angle <= 0 or angle >= 180: # 三点共线, 找下一个夹角的点
                    continue
                
                # 三点确实可以组成一个三角形
                ## check whether there is any vertices within this triangle
                in_triangle = False 
                for j in range(len(vertices)):
                    if inside_triangle(vertices[j], prev_vertex.position, curr_vertex.position, next_vertex.position) and \
                       eq_npfloat32(vertices[j].position, prev_vertex.position) and \
                       eq_npfloat32(vertices[j].position, curr_vertex.position) and \
                       eq_npfloat32(vertices[j].position, next_vertex.position):
                       # 当前点确实在所构成的三角形内
                        in_triangle = True 
                        break 
                if in_triangle:
                    continue
                
                for j in range(len(vertices)):
                    if eq_npfloat32(vertices[j].position, prev_vertex.position):
                        indices.append(j)
                    if eq_npfloat32(vertices[j].position, next_vertex.position):
                        indices.append(j)
                    if eq_npfloat32(vertices[j].position, temp_vertex):
                        indices.append(j)

                for j in range(len(tvertices)):
                    if eq_npfloat32(tvertices[j].position, curr_vertex.position):
                        del tvertices[j]  
                        break

                i -= 1

            if len(indices) == 0 or len(tvertices) == 0:
                break
                
        return indices


def eq_npfloat32(a, b):
    if a.shape == b.shape:
        d = np.abs(a-b)
        shape = a.shape
        i = 1
        for c in shape:
            i *= c 
        if (d<1e-6).sum() == i:
            return True 
    return False




