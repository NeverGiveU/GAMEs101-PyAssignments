from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 


class Texture(object):
    def __init__(self, tex_path):
        img = Image.open(tex_path).convert("RGB")
        self.arr = np.array(img)
        self.height, self.width = self.arr.shape[:2]

    def get_color(self, u, v):
        '''
        @param
            u, v --np.array --dtype=np.float32 --range=[0., 1.] --shape=(h,w)
        @return 
            color --np.array --shape=(3,)
        '''
        # u = min(1., max(0., u))
        # v = min(1., max(0., v))
        u = np.clip(u, 0., 1.)
        v = np.clip(v, 0., 1.)

        x = u*self.width
        y = (1-v)*self.height
        
        colors = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.float32)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                xx = min(self.width-1, int(x[i,j]))
                yy = min(self.height-1, int(y[i,j]))
                colors[i,j,:] = self.arr[yy, xx, :]
        return colors

    
    