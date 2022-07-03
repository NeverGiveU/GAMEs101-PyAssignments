import argparse
import os 
import matplotlib.pyplot as plt 
import numpy as np 
from Application import AppConfig, Application 
from tqdm import tqdm  
import cv2 
import imageio


if __name__ == "__main__":
    config = AppConfig()
    app = Application(config)

    app.initialize()
    for i in tqdm(range(1024)):
        app.render(index=i)

    ## 保存为视频
    # save_name = config.save_name + ".avi"
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fps = 30
    # width = app.width
    # height = app.height
    # fh = cv2.VideoWriter(save_name, fourcc, fps, (width, height))
    # for i in tqdm(range(1024)):
    #     pth = os.path.join(config.save_name, "%04d.png"%i)
    #     arr = cv2.imread(pth)
    #     fh.write(arr)
    # fh.release()

    ## 保存为 gif
    save_name = config.save_name + ".gif"
    frames = []
    for i in tqdm(range(1024)):
        pth = os.path.join(config.save_name, "%04d.png"%i)
        arr = imageio.imread(pth)
        frames.append(arr)
    imageio.mimsave(save_name, frames, fps=30)





