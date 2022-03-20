import numpy as np  
import Triangle


class vertex_buf_id(object):
    def __init__(self, _id):
        self._id = _id


class indice_buf_id(object):
    def __init__(self, _id):
        self._id = _id


COLOR = 1
DEPTH = 2


class Rasterizer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.color_buf = np.zeros((self.height, self.width, 3), dtype=np.float32)
        self.depth_buf = np.array([[float("inf")]*self.width for _ in range(self.height)], dtype=np.float32)
        
        self.next_id = 0  
        self.vertex_buf = {}
        self.indice_buf = {}

        self.Mmatrix = None
        self.Vmatrix = None 
        self.Pmatrix = None

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

    def clear(self, signal):
        if signal&COLOR == COLOR:
            self.color_buf = np.zeros((self.height, self.width, 3), dtype=np.float32)
        if signal&DEPTH == DEPTH:
            self.depth_buf = np.array([[-float("inf")]*self.width for _ in range(self.height)], dtype=np.float32)
    
    def set_Mmatrix(self, Mmatrix):
        self.Mmatrix = Mmatrix

    def set_Vmatrix(self, Vmatrix):
        self.Vmatrix = Vmatrix

    def set_Pmatrix(self, Pmatrix):
        self.Pmatrix = Pmatrix
    
    def rasterize(self, v_id, i_id, ptype):
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

        # 获取节点和连接索引
        vertices = self.vertex_buf[v_id._id]
        indices = self.indice_buf[i_id._id]

        self.MVPmatrix = np.dot(self.Pmatrix, np.dot(self.Vmatrix, self.Mmatrix))
        
        f1 = (100-0.1)/2.0
        f2 = (100+0.1)/2.0

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
            # 坐标变换
            primitivity = np.dot(self.MVPmatrix, primitivity.T).T # (3,4)
            # 归一化齐次坐标
            primitivity = primitivity / primitivity[:, 3:4]

            # 投影到屏幕
            primitivity[:, 0:1] = 0.5*self.width*(primitivity[:, 0:1]+1.0)  # [-1, 1] -> [0, width]
            primitivity[:, 1:2] = 0.5*self.height*(primitivity[:, 1:2]+1.0) # [-1, 1] -> [0, height]
            # 至于 z, 在本题中随便都可以?
            primitivity[:, 2:3] = primitivity[:, 2:3]*f1+f2 

            ## 赋值到三角形单元中
            t = Triangle.Triangle()

            t.set_vertex(0, primitivity[0, :3])
            t.set_vertex(1, primitivity[1, :3])
            t.set_vertex(2, primitivity[2, :3])

            t.set_color(0, 255.0, 0.0, 0.0)
            t.set_color(1, 0.0, 255.0, 0.0)
            t.set_color(2, 0.0, 0.0, 255.0)

            self.rasterize_wireframe(t)

    def rasterize_wireframe(self, t):
        '''
        @param
            t --type=Triangle
        '''
        self.draw_line(t.a()[0], t.b()[0], t.get_color(t.a()[1]), t.get_color(t.b()[1]))
        self.draw_line(t.b()[0], t.c()[0], t.get_color(t.b()[1]), t.get_color(t.c()[1]))
        self.draw_line(t.c()[0], t.a()[0], t.get_color(t.c()[1]), t.get_color(t.a()[1]))

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

        # 专为整型
        if color1 is None:
            color1 = np.array([255, 255, 255])
        if color2 is None:
            color2 = np.array([255, 255, 255])
        # color = np.array([255, 255, 255]) # 仅考虑白色

        x = y = 0.     # 起始点, 中间点的坐标
        dx = dy = 0.
        dx1 = dy1 = 0.
        px = py = 0.
        xs = ys = 0.
        xe = ye = 0.   # 终点坐标
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

        if dy1 <= dx1:  # 属于扁平的, 即倾斜度 <= 45°
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
                # 计算颜色
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

        else:           # 属于瘦高的, 即倾斜度 > 45°
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
                # 计算颜色
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
            # ind = (self.height-1-y)*self.width+x # 考虑到数组中一般原点在左上角, 我们习惯是在左下角
            # self.color_buf[ind] = color
            self.color_buf[self.height-1-y, x, :] = color
    
    def get_screenshot(self):
        return self.color_buf.astype(np.uint8)
