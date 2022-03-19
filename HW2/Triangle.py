import numpy as np  


class Triangle(object):
    def __init__(self):
        self.vertices = [None]*3 # 按顺时针方向的三角形三个顶点的初始坐标
        self.colors = [None]*3   # 每个顶点的颜色 RGB 值
        self.textures = [None]*3 # 每个顶点的纹理 (uv 坐标)
        self.normals = [None]*3  # 每个顶点的法向量

    ## 按顺序返回每个顶点的坐标值
    def a(self):
        return self.vertices[0], 0

    def b(self):
        return self.vertices[1], 1

    def c(self):
        return self.vertices[2], 2

    def set_vertex(self, ind, vertex):
        self.vertices[ind] = vertex

    def set_color(self, ind, r, g, b):
        r = min(max(0., r), 255.)
        g = min(max(0., g), 255.)
        b = min(max(0., b), 255.)
        self.colors[ind] = np.array([r, g, b], dtype=np.float32)

    def get_color(self, ind):
        if 0 <= ind < len(self.colors):
            return self.colors[ind]
