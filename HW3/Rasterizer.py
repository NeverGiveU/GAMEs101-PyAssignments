import numpy as np  
import Triangle
import random
from Texture import Texture 
# from Shader import FragmentShaderPayload
from Shader import *
import os  


class vertex_buf_id(object):
    def __init__(self, _id):
        self._id = _id


class indice_buf_id(object):
    def __init__(self, _id):
        self._id = _id


class color_buf_id(object):
    def __init__(self, _id):
        self._id = _id 


COLOR = 1
DEPTH = 2


class Rasterizer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.color_buf = np.zeros((self.height, self.width, 3), dtype=np.float32)
        # self.depth_buf = np.array([[float("inf")]*self.width for _ in range(self.height)], dtype=np.float32)
        self.depth_buf = np.array([[1e8]*self.width for _ in range(self.height)], dtype=np.float32)
        
        self.next_id = 0  
        self.vertex_buf = {}
        self.indice_buf = {}
        self.vcolor_buf = {} # store colors for each vertex

        self.Mmatrix = None
        self.Vmatrix = None 
        self.Pmatrix = None

        self.texture = None  # type=Texture
        self.shader_type = "displacement"#"normal"#"bump"#"phong"#"texture"#
        self.obj_path = ""

    def get_next_id(self):
        self.next_id += 1
        return self.next_id - 1

    def load_vertices(self, vertices):
        '''
        @param
            vertices --np.array --shape=(N,3)
        @return
            v_id --vertex_buf_id 
        '''
        _id = self.get_next_id()
        self.vertex_buf[_id] = vertices
        return vertex_buf_id(_id)
    
    def load_indices(self, indices):
        '''
        @param
            indices --np.array --shape=(N//3,3) --dtype=np.uint8
        @return
            i_id --indice_buf_id 
        '''
        _id = self.get_next_id()
        self.indice_buf[_id] = indices
        return indice_buf_id(_id)

    def load_colors(self, colors):
        '''
        @param
            colors --np.array --shape=(N,3) --dtype=np.float32
        @return 
            c_id --colors_buf_id
        '''
        _id = self.get_next_id()
        self.vcolor_buf[_id] = colors 
        return color_buf_id(_id)


    def clear(self, signal):
        if signal&COLOR == COLOR:
            self.color_buf = np.zeros((self.height, self.width, 3), dtype=np.float32)
        if signal&DEPTH == DEPTH:
            # self.depth_buf = np.array([[float("inf")]*self.width for _ in range(self.height)], dtype=np.float32)
            self.depth_buf = np.array([[1e8]*self.width for _ in range(self.height)], dtype=np.float32)
        
    
    def set_Mmatrix(self, Mmatrix):
        self.Mmatrix = Mmatrix

    def set_Vmatrix(self, Vmatrix):
        self.Vmatrix = Vmatrix

    def set_Pmatrix(self, Pmatrix):
        self.Pmatrix = Pmatrix

    def set_texture_path(self, tex_path):
        self.texture = Texture(tex_path)
    
    def rasterize(self, v_id, i_id, c_id, ptype):
        '''
        @param
            v_id --type=vertex_buf_id
            i_id --type=indice_buf_id
            ptype --str
        @return
            None
        '''
        assert ptype == "Triangle", "Drawing primitives other than triangle is not implemented yet!"
        assert self.Mmatrix is not None and self.Vmatrix is not None and self.Pmatrix is not None

        # ???????????????????????????
        vertices = self.vertex_buf[v_id._id]
        indices = self.indice_buf[i_id._id]
        colors = self.vcolor_buf[c_id._id]

        self.MVPmatrix = np.dot(self.Pmatrix, np.dot(self.Vmatrix, self.Mmatrix))
        f1 = (50-0.1)/2.0
        f2 = (50+0.1)/2.0

        for indice in indices:
            '''
            indice --np.array --shape=(3,) --dtype=np.uint8
            '''
            primitivity = np.concatenate([
                np.array([vertices[indice[0]],
                          vertices[indice[1]],
                          vertices[indice[2]]]),
                np.ones((3, 1), dtype=np.float32)
            ], axis=1) # (3,4)
            # ????????????
            primitivity = np.dot(self.MVPmatrix, primitivity.T).T # (3,4)
            # ?????????????????????
            primitivity = primitivity / primitivity[:, 3:4]

            # ???????????????
            primitivity[:, 0:1] = 0.5*self.width*(primitivity[:, 0:1]+1.0)  # [-1, 1] -> [0, width]
            primitivity[:, 1:2] = 0.5*self.height*(primitivity[:, 1:2]+1.0) # [-1, 1] -> [0, height]
            # ?????? z, ????????????????????????????
            primitivity[:, 2:3] = primitivity[:, 2:3]*f1+f2 

            ## ???????????????????????????
            t = Triangle.Triangle()

            t.set_vertex(0, primitivity[0, :3])
            t.set_vertex(1, primitivity[1, :3])
            t.set_vertex(2, primitivity[2, :3])

            t.set_color(0, colors[indice[0]][0], colors[indice[0]][1], colors[indice[0]][2])
            t.set_color(1, colors[indice[1]][0], colors[indice[1]][1], colors[indice[1]][2])
            t.set_color(2, colors[indice[2]][0], colors[indice[2]][1], colors[indice[2]][2])

            # self.rasterize_wireframe(t)
            self.rasterize_triangle(t)

    def rasterize_wireframe(self, t):
        '''
        @param
            t --type=Triangle
        '''
        self.draw_line(t.a()[0], t.b()[0], t.get_color(t.a()[1]), t.get_color(t.b()[1]))
        self.draw_line(t.b()[0], t.c()[0], t.get_color(t.b()[1]), t.get_color(t.c()[1]))
        self.draw_line(t.c()[0], t.a()[0], t.get_color(t.c()[1]), t.get_color(t.a()[1]))

    def rasterize_triangle(self, t, viewspace_pos):
        # print(dir(t))
        # print("------------------The current triangle")
        # print(t.vertices)
        # print([c/255. for c in t.colors])
        # print(t.textures)
        # print(viewspace_pos)
        '''
        @param
            t --type=Triangle
        '''
        (x1, y1, z1, w1), _id1 = t.a()
        (x2, y2, z2, w2), _id2 = t.b()
        (x3, y3, z3, w3), _id3 = t.c()

        # x1, y1 = int(x1), int(y1)
        # x2, y2 = int(x2), int(y2)
        # x3, y3 = int(x3), int(y3)

        x_min = int(min(x1, x2, x3))
        x_max = int(max(x1, x2, x3))
        y_min = int(min(y1, y2, y3))
        y_max = int(max(y1, y2, y3))

        ## get the bounding box of the current triangle

        if x_max <= 0 or y_max <= 0 or x_min >= self.width or y_min >= self.height:
            # if totally outside the screen, just ignore it.
            return 

        vertices = np.array([[x1, y1],
                             [x2, y2],
                             [x3, y3]])
        
        """ To slow
        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                f = self.inside_triangle(x, y, vertices)
                ## ????????????????????? z 
                alpha, beta, gamma = self.compute_Barycentric2D(x, y, vertices)
                w_reciprocal = 1.0/(alpha + beta + gamma)
                z = alpha*z1 + beta*z2 + gamma*z3

                color = alpha/(alpha+beta+gamma)*t.get_color(_id1) + beta/(alpha+beta+gamma)*t.get_color(_id2) + gamma/(alpha+beta+gamma)*t.get_color(_id3)
                z = w_reciprocal*z 

                if f:
                    if self.depth_buf[x, y] < z:
                        self.set_pixel(x, y, color)
                        self.depth_buf[x, y] = z 
        """

        ## ????????????
        ys, xs = np.mgrid[y_min:y_max:complex("{}j".format(y_max-y_min+1)), x_min:x_max:complex("{}j".format(x_max-x_min+1))]
        alphas, betas, gammas = self.compute_Barycentric2D_parallel(xs+0.5, ys+0.5, vertices)
        # print(xs[0,0], ys[0,0], alphas[0,0], betas[0,0], gammas[0,0])
        # print(gammas)
        ws = 1.0/(alphas/(w1+1e-8)+betas/(w2+1e-8)+gammas/(w3+1e-8)+1e-8)
        zs = alphas*z1 + betas*z2 + gammas*z3
        zs = ws*zs 
        # print(zs)
        ## ?????????????????????
        alphas_betas_gammas = (alphas+betas+gammas)+1e-8
        alphas = alphas[:, :, np.newaxis]#(alphas/alphas_betas_gammas)[:, :, np.newaxis] # (H, W, 1)
        betas = betas[:, :, np.newaxis]#(betas/alphas_betas_gammas)[:, :, np.newaxis]
        gammas = gammas[:, :, np.newaxis]#(gammas/alphas_betas_gammas)[:, :, np.newaxis]

        colors         = alphas*t.colors     [0][np.newaxis, np.newaxis, :] + betas*t.colors     [1][np.newaxis, np.newaxis, :] + gammas*t.colors     [2][np.newaxis, np.newaxis, :]
        normals        = alphas*t.normals    [0][np.newaxis, np.newaxis, :] + betas*t.normals    [1][np.newaxis, np.newaxis, :] + gammas*t.normals    [2][np.newaxis, np.newaxis, :]
        tex_coords     = alphas*t.textures   [0][np.newaxis, np.newaxis, :] + betas*t.textures   [1][np.newaxis, np.newaxis, :] + gammas*t.textures   [2][np.newaxis, np.newaxis, :]
        viewspace_poses= alphas*viewspace_pos[0][np.newaxis, np.newaxis, :] + betas*viewspace_pos[1][np.newaxis, np.newaxis, :] + gammas*viewspace_pos[2][np.newaxis, np.newaxis, :]

        ## ????????????
        # ????????????????????????????????????
        normals = normals / (np.sqrt(np.power(normals, 2).sum(axis=2))[:, :, np.newaxis]+1e-6)
        fragment_shader_payload = FragmentShaderPayload(viewspace_poses, colors, normals, tex_coords, self.texture)
        pixel_colors = self.fragment_shader(fragment_shader_payload)
        colors = pixel_colors

        # a map
        M = self.inside_triangle_parallel(xs, ys, vertices) # ?????????????????????

        # ??????????????????????????????
        ct = max(-y_min, 0)
        cd = max(y_max-(self.height-1), 0)
        cl = max(-x_min, 0)
        cr = max(x_max-(self.width-1), 0)

        M = M[ct:M.shape[0]-cd, cl:M.shape[1]-cr]
        colors = colors[ct:colors.shape[0]-cd, cl:colors.shape[1]-cr, :]
        zs = zs[ct:zs.shape[0]-cd, cl:zs.shape[1]-cr]

        # ?????? 
        dt = max(0, y_min)
        db = max(self.height-y_max-1, 0)
        dl = max(0, x_min) 
        dr = max(self.width-x_max-1, 0)
        # print(zs.shape)
        M = np.pad(M, ((dt, db), (dl, dr)), "constant", constant_values=0.)
        colors = np.pad(colors, ((dt, db), (dl, dr), (0, 0)), "constant", constant_values=0.)
        zs = np.pad(zs, ((dt, db), (dl, dr)), "constant", constant_values=0.)

        valid_M = (zs < self.depth_buf)
        valid_M = ((M*valid_M)[:, :, np.newaxis]).astype(np.float32)
        # import matplotlib.pyplot as plt 
        # plt.imshow(valid_M[:, :, 0])
        # plt.show()

        self.color_buf = self.color_buf*(1-valid_M)+colors*valid_M
        self.depth_buf = self.depth_buf*(1-valid_M[:, :, 0])+zs*valid_M[:, :, 0]

    def compute_Barycentric2D_parallel(self, x, y, vertices):
        '''
        @param
            x --np.array --shape=(H,W)
            y --np.array --shape=(H,W)

            vertices --np.array --shape=(3,2)
        '''
        c1 = (x*(vertices[1][1]-vertices[2][1]) + (vertices[2][0]-vertices[1][0])*y + vertices[1][0]*vertices[2][1]-vertices[2][0]*vertices[1][1]) / \
             (vertices[0][0]*(vertices[1][1]-vertices[2][1]) + (vertices[2][0]-vertices[1][0])*vertices[0][1] + vertices[1][0]*vertices[2][1] - vertices[2][0]*vertices[1][1] + 1e-6)
        c2 = (x*(vertices[2][1]-vertices[0][1]) + (vertices[0][0]-vertices[2][0])*y + vertices[2][0]*vertices[0][1]-vertices[0][0]*vertices[2][1]) / \
             (vertices[1][0]*(vertices[2][1]-vertices[0][1]) + (vertices[0][0]-vertices[2][0])*vertices[1][1] + vertices[2][0]*vertices[0][1] - vertices[0][0]*vertices[2][1] + 1e-6)
        c3 = (x*(vertices[0][1]-vertices[1][1]) + (vertices[1][0]-vertices[0][0])*y + vertices[0][0]*vertices[1][1]-vertices[1][0]*vertices[0][1]) / \
             (vertices[2][0]*(vertices[0][1]-vertices[1][1]) + (vertices[1][0]-vertices[0][0])*vertices[2][1] + vertices[0][0]*vertices[1][1] - vertices[1][0]*vertices[0][1] + 1e-6)
        return c1, c2, c3

    def compute_Barycentric2D(self, x, y, vertices):
        c1 = (x*(vertices[1][1]-vertices[2][1]) + (vertices[2][0]-vertices[1][0])*y + vertices[1][0]*vertices[2][1]-vertices[2][0]*vertices[1][1]) / \
             (vertices[0][0]*(vertices[1][1]-vertices[2][1]) + (vertices[2][0]-vertices[1][0])*vertices[0][1] + vertices[1][0]*vertices[2][1] - vertices[2][0]*vertices[1][1])
        c2 = (x*(vertices[2][1]-vertices[0][1]) + (vertices[0][0]-vertices[2][0])*y + vertices[2][0]*vertices[0][1]-vertices[0][0]*vertices[2][1]) / \
             (vertices[1][0]*(vertices[2][1]-vertices[0][1]) + (vertices[0][0]-vertices[2][0])*vertices[1][1] + vertices[2][0]*vertices[0][1] - vertices[0][0]*vertices[2][1])
        c3 = (x*(vertices[0][1]-vertices[1][1]) + (vertices[1][0]-vertices[0][0])*y + vertices[0][0]*vertices[1][1]-vertices[1][0]*vertices[0][1]) / \
             (vertices[2][0]*(vertices[0][1]-vertices[1][1]) + (vertices[1][0]-vertices[0][0])*vertices[2][1] + vertices[0][0]*vertices[1][1] - vertices[1][0]*vertices[0][1])
        return c1, c2, c3

    def inside_triangle(self, x, y, vertices):
        '''
        @param
            x, t --pixel_coordinate
            vertices --np.array --shape=(3,2)
        '''
        AB_BC_CA = vertices[[1,2,0], :] - vertices
        AD_BD_CD = np.array([[x, y]]) - vertices
        ABxAD_BCxBD_CAxCD = AB_BC_CA*AD_BD_CD[:, ::-1]
        ABxAD_BCxBD_CAxCD = ABxAD_BCxBD_CAxCD[:, 0] - ABxAD_BCxBD_CAxCD[:, 1]

        if (ABxAD_BCxBD_CAxCD>=0).sum() == 3 or (ABxAD_BCxBD_CAxCD<=0).sum() == 3:
            return True
        return False

    def inside_triangle_parallel(self, x, y, vertices):
        '''
        @param
            x --np.array --shape=(H,W)
            y --np.array --shape=(H,W)
        '''
        AB_BC_CA = vertices[[1,2,0], :] - vertices # (3,2)
        Ds = np.concatenate((x[:, :, np.newaxis, np.newaxis], y[:, :, np.newaxis, np.newaxis]), axis=3)
        AD_BD_CD = Ds - vertices # (H,W,3,2)
        ABxAD_BCxBD_CAxCD = AB_BC_CA[np.newaxis, np.newaxis, :, ::-1]*AD_BD_CD
        ABxAD_BCxBD_CAxCD = ABxAD_BCxBD_CAxCD[:, :, :, 0] - ABxAD_BCxBD_CAxCD[:, :, :, 1]

        M1 = (ABxAD_BCxBD_CAxCD>=0).sum(axis=2) == 3
        M2 = (ABxAD_BCxBD_CAxCD<=0).sum(axis=2) == 3
        M = np.logical_or(M1, M2)

        # import matplotlib.pyplot as plt 
        # plt.imshow(M)
        # plt.show()
        return M

    def draw_line(self, vertex1, vertex2, color1=None, color2=None):
        '''
        @param
            vertex1 --type=np.array --shape=(3,)
            vertex2 --type=np.array --shape=(3,)
        '''
        x1, y1 = vertex1[:2]
        x2, y2 = vertex2[:2]

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # ????????????
        if color1 is None:
            color1 = np.array([255, 255, 255])
        if color2 is None:
            color2 = np.array([255, 255, 255])
        # color = np.array([255, 255, 255]) # ???????????????

        x = y = 0.     # ?????????, ??????????????????
        dx = dy = 0.
        dx1 = dy1 = 0.
        px = py = 0.
        xs = ys = 0.
        xe = ye = 0.   # ????????????
        i = 0 

        ##
        dx = x2-x1
        dy = y2-y1
        dx1 = abs(dx)
        dy1 = abs(dy)
        px = 2*dy1-dx1
        py = 2*dx1-dy1

        '''
        y                    y                    y                    y                   
        ^                    ^                    ^                    ^                    
        |            *v2     |            *v1     |   *v2              |   *v1              
        |                    |                    |                    |                    
        |                    |                    |                    |                    
        |                    |                    |                    |                    
        |  *v1               |  *v2               |            *v1     |            *v2     
        +---------------->x  +---------------->x  +---------------->x  +---------------->x  
                (1)                   (2)                  (3)                 (4)
        '''

        if dy1 <= dx1:  # ???????????????, ???????????? <= 45??
            if dx >= 0: # (1)(4)
                x = x1 
                y = y1 
                xe = x2 # end
                xs = x1 # begin
            else:       # (2)(3)
                x = x2 
                y = y2
                xe = x1 
                xs = x2
                color1, color2 = color2, color1
            
            self.set_pixel(x, y, color1)
            while x < xe:
                x += 1
                # ????????????
                alpha = (x-xs)/dx1
                color = (alpha*color2 + (1-alpha)*color1).astype(np.uint8)

                if px < 0:
                    px = px + 2*dy1 # keep y unchanged
                else:
                    if dx < 0 and dy < 0 or dx > 0 and dy > 0: # (1)(2)
                        y += 1
                    else:
                        y -= 1
                    px = px + 2*(dy1-dx1)
                self.set_pixel(x, y, color)

        else:           # ???????????????, ???????????? > 45??
            if dy >= 0: # (1)(3)
                x = x1 
                y = y1
                ye = y2
                ys = y1
            else:       # (2)(4)
                x = x2 
                y = y2 
                ye = y1 
                ys = y2 
                color1, color2 = color2, color1

            self.set_pixel(x, y, color1)
            while y < ye:
                y += 1
                # ????????????
                alpha = (y-ys)/dy1
                color = (alpha*color2 + (1-alpha)*color1).astype(np.uint8)

                if py <= 0:
                    py = py + 2*dx1 # keep x unchanged
                else:
                    if dx < 0 and dy < 0 or dx > 0 and dy > 0:
                        x += 1
                    else:
                        x -= 1
                    py = py + 2*(dx1- dy1)
                self.set_pixel(x, y, color)

    def set_pixel(self, x, y, color):
        if 0 <= x < self.width-1 and 0 <= y < self.height-1:
            # ind = (self.height-1-y)*self.width+x # ??????????????????????????????????????????, ???????????????????????????
            # self.color_buf[ind] = color
            self.color_buf[self.height-1-y, x, :] = color
    
    def get_screenshot(self):
        return self.color_buf.astype(np.uint8)

    def render(self, triangles):
        self.MVPmatrix = np.dot(self.Pmatrix, np.dot(self.Vmatrix, self.Mmatrix))
        # print("MVP matrix:")
        # print(self.MVPmatrix)
        f1 = (50-0.1)/2.0
        f2 = (50+0.1)/2.0

        for triangle in triangles:
            vertices = np.array(triangle.vertices)

            mm = np.dot(self.Vmatrix, np.dot(self.Mmatrix, vertices.T)).T # (3,4)
            viewspace_pos = mm[:, :3]                                     # (3,3)
            
            # ????????????
            vertices = np.dot(self.MVPmatrix, vertices.T).T               # (3,4)
            # ?????????????????????
            vertices = vertices / vertices[:, 3:4]
            
            ## ?????????????????? normal ????????????????????????
            inv_MVmatrix = np.linalg.inv(np.dot(self.Vmatrix, self.Mmatrix)).T
            normals = np.dot(inv_MVmatrix, 
                np.concatenate([np.array(triangle.normals), np.zeros((3,1), dtype=np.float32)], axis=1).T
            ).T

            ## Viewport transformation# ????????????: ???????????????
            vertices[:, 0:1] = 0.5*self.width*(vertices[:, 0:1]+1.0)  # [-1, 1] -> [0, width]
            vertices[:, 1:2] = 0.5*self.height*(vertices[:, 1:2]+1.0) # [-1, 1] -> [0, height]
            vertices[:, 2:3] = vertices[:, 2:3]*f1+f2 

            triangle_cached = Triangle.Triangle()
            # r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            r, g, b = 148., 121., 92.
            for _id in range(3):
                # r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                triangle_cached.set_vertex(_id, vertices[_id])
                triangle_cached.set_normal(_id, normals[_id][:3])
                triangle_cached.set_color(_id, r, g, b)
            triangle_cached.textures = triangle.textures

            # Also pass view space vertice position
            self.rasterize_triangle(triangle_cached, viewspace_pos)

            ## some notes here
            '''
            for the transformation of a vertex, just

            V.dot(M).dot(vertex) -> word-coordinate of the vertex


            but, fot the transformation of the normal, we need 

            ((V.dot(M))^-1)^T.dot(normal)

            PLS refer to: https://blog.csdn.net/u010006851/article/details/52929100
            '''
            # break


    def fragment_shader(self, payload):
        '''
        @param
            payload --type=FragmentShaderPayload
        @return
        '''
        if self.shader_type == "normal":
            return normal_fragment_shader(payload)
        elif self.shader_type == "texture":
            self.set_texture_path(self.obj_path.replace(os.path.basename(self.obj_path), "texture.png"))
            payload.texture = self.texture
            return texture_fragment_shader(payload)
        elif self.shader_type == "phong":
            return phong_fragment_shader(payload)
        elif self.shader_type == "bump":
            return bump_fragment_shader(payload)
        elif self.shader_type == "displacement":
            return displacement_fragment_shader(payload)
        else:
            return