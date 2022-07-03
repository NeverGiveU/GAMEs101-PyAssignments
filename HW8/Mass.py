import numpy as np  


class Mass(object):
    def __init__(self, 
                 position, # --type=np.array --shape=(2,) --help="质点的初始位置"
                 mass,     # --type=float
                 pinned,   # --type=boolean --help="该点是否可动"
                 ):
        self.start_position = position.copy() # 初始位置
        self.position = position.copy()       # 当前位置
        self.last_position = position.copy()  # 上一时刻的位置

        self.velocity = np.zeros(2, dtype=np.float32)
        self.forces = np.zeros(2, dtype=np.float32)

