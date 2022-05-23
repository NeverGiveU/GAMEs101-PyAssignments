import numpy as np  
import matplotlib.pyplot as plt 
from Triangle import MeshTriangle
from Scene import Scene 
from Light import Light
from Renderer import Renderer
import time 


if __name__ == "__main__":
    scene = Scene(1280, 960)
    # scene = Scene(320, 240)
    # scene = Scene(80, 60)

    ## Object
    bunny = MeshTriangle("./bunny.obj")
    scene.add_object(bunny)

    ## Light(s)
    light1 = Light(np.array([-20., 70., 20.]), np.array([1., 1., 1.]))
    light2 = Light(np.array([ 20., 70., 20.]), np.array([1., 1., 1.]))
    scene.add_light(light1)
    scene.add_light(light2)

    scene.buildBVH() ## 对场景中的对象再建立一层 BVH

    ## Renderer
    renderer = Renderer()
    stime = time.time()
    renderer.render(scene)
    etime = time.time()
    print(etime-stime)
    
    ## 
    plt.imshow(renderer.frame_buffer)
    plt.show()