import numpy as np 
import math 
from Rope import Rope 
import cv2 
from PIL import Image
import os 


class AppConfig(object):
    def __init__(self):
        self.mass = 1.
        self.ks = 100. # 弹簧的弹性系数
        self.gravity = np.array([0., 1.])
        self.steps_per_frame = 64
        self.save_name = "sample"
        if os.path.exists(self.save_name) is not True:
            os.mkdir(self.save_name)


class Application(object):
    def __init__(self, config):
        self.rope_Euler = None
        self.rope_Verlet = None

        self.width = 480
        self.height = 480

        self.config = config

    def initialize(self):
        self.rope_Euler = Rope(np.array([240., 0.]),
                               np.array([400., 0.]),
                               5,
                               self.config.mass, self.config.ks, [0])
        self.rope_Verlet= Rope(np.array([240., 0.]),
                               np.array([400., 0.]),
                               10,
                               self.config.mass, self.config.ks, [0])

    def render(self, index=0):
        # 将一帧拆分为多个步骤 (1 seconds -> 1*24 frames -> 1*24*? Δt)
        for i in range(self.config.steps_per_frame):
            self.rope_Euler.simulate_Euler(1/self.config.steps_per_frame, self.config.gravity)
            self.rope_Verlet.simulate_Euler(1/self.config.steps_per_frame, self.config.gravity)
        
        ropes = [self.rope_Euler, self.rope_Verlet]
        colors = [np.array([0.0, 0.0, 1.0]),
                  np.array([0.0, 1.0, 0.0])]
        
        arr = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for rope, color in zip(ropes, colors):
            masses = rope.masses
            for m in masses:
                x, y = m.pos
                cv2.circle(arr, (int(x), int(y)), 10, color, 0)
            springs = rope.springs
            for s in springs:
                x0, y0 = s.m1.pos 
                x1, y1 = s.m2.pos
                cv2.line(arr, (int(x0), int(y0)), (int(x1), int(y1)), color, 2, 4)
            # break
        Image.fromarray((arr*255).astype(np.uint8)).save(os.path.join(self.config.save_name, "%04d.png"%index))



