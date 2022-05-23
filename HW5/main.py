import numpy as np  
import matplotlib.pyplot as plt 
from Scene import Scene
from Object import Sphere, MeshTriangle, Light
from Renderer import Renderer
import time 


if __name__ == "__main__":
    scene = Scene(1280, 960)
    # scene = Scene(320, 240) # for debug
    
    ## Add object(s)
    sph1 = Sphere(np.array([-1., 0., -12.]), 2.)
    sph1.material_type = "DIFFUSE_AND_GLOSSY"
    sph1.color_diffuse = np.array([.6, .7, .8])

    sph2 = Sphere(np.array([.5, -.5, -8.]), 1.5)
    sph2.ior = 1.5 # 折射率
    sph2.material_type = "REFLECTION_AND_REFRACTION"

    scene.add_object(sph1)
    scene.add_object(sph2)
    
    ## Ground
    vertices = np.array([
        [-5., -3., -6.],
        [5., -3., -6.],
        [5., -3., -16.],
        [-5., -3., -16.]
    ])
    indices = [0, 1, 3, 1, 2, 3]
    st = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1]
    ])
    mesh = MeshTriangle(vertices, indices, len(indices)//3, st)
    mesh.material_type = "DIFFUSE_AND_GLOSSY"

    scene.add_object(mesh)

    ## Light(s)
    light1 = Light(np.array([-20., 70., 20.]), np.array([.5, .5, .5]))
    light2 = Light(np.array([30., 50., -12.]), np.array([.5, .5, .5]))

    scene.add_light(light1)
    scene.add_light(light2)

    renderer = Renderer()
    stime = time.time()
    renderer.render(scene)
    etime = time.time()
    print("Time Cost: {:.4f}".format(etime-stime))

    plt.imshow(renderer.frame_buffer)
    plt.show()