import numpy as np 


class BBox3D(object):
    def __init__(self, min_vertex=None, max_vertex=None):
        '''
        @param
            `min_vertex` --type=np.array --shape=(3,)
            `max_vertex` --type=np.array --shape=(3,)            
        '''
        if min_vertex is not None and max_vertex is not None:
            self.min_vertex = np.minimum(min_vertex, max_vertex)
            self.max_vertex = np.maximum(min_vertex, max_vertex)
        else:
            self.min_vertex = np.array([float("inf"), float("inf"), float("inf")], dtype=np.float32)
            self.max_vertex = np.array([-float("inf"), -float("inf"), -float("inf")], dtype=np.float32)

    def centroid(self):
        '''
        @return
            `ctr` --type=np.array --shape=(3,)
        '''
        return self.min_vertex * .5 + self.max_vertex * .5

    def diagonal(self):
        return self.max_vertex - self.min_vertex

    def max_extent(self):
        '''
        @help
            --description="返回较长的轴 // heuristic1"
        '''
        dif = self.diagonal()
        if dif[0] > dif[1] and dif[1] > dif[2]:
            return 0 ## x-axis
        elif dif[1] > dif[2]:
            return 1 
        else:
            return 2 

    def intersect(self, ray, inv_direction, dir_is_neg):
        '''
        @help
            --description="预判光线 ray 与包围盒是否相交"
        @param
            `ray` --type=Ray
            `inv_direction` --type=np.array<np.float32> --shape=(3,)
            `dir_is_neg` --type=np.array<np.uint8> --shape=(3,)
        '''
        original = ray.original
        t_enter = -float("inf")
        t_exit = float("inf")
        # 考虑每个维度
        for i in range(3):
            m = (self.min_vertex[i]-original[i]) * inv_direction[i]
            M = (self.max_vertex[i]-original[i]) * inv_direction[i] 
            # 乘上一个权重, 关键是 sign, 因为如果方向在这个维度指向负方向, 
            # 那么包围盒减去原点的坐标就是负数, 但是我们希望统一这个差额是正数
            if dir_is_neg[i] < 1:
                # 如果方向在这个维度指向负方向, 相对距离的远近需要调换顺序
                m, M = M, m 
            t_enter = max(m, t_enter)
            t_exit = min(M, t_exit)

        return t_enter < t_exit and t_exit >= 0


def bbox3D_union_bbox3D(bbox1, bbox2):
    return BBox3D(np.minimum(bbox1.min_vertex, bbox2.min_vertex),
                  np.maximum(bbox1.max_vertex, bbox2.max_vertex))

def bbox3D_union_vertex(bbox, vertex):
    return BBox3D(np.minimum(bbox.min_vertex, vertex), np.maximum(bbox.max_vertex, vertex))
