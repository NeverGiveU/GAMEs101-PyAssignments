import numpy as np 
import math 
import Triangle
import Rasterizer
import argparse
import matplotlib.pyplot as plt 

from OBJloader import *


def get_Mmatrix(angle_x, angle_y, angle_z):
    '''
    @param
        angle_x, angle_y, angle_z --float
    @return
        Mmatrix --np.array --shape=(4,4)

    ## 用于对单个模型做变换，相当于摆 pose
    '''
    angle_x = angle_x/180.0 * _PI
    Mmatrix_X = np.array([[1., 0., 0., 0.],
                          [0., math.cos(angle_x), -math.sin(angle_x), 0.],
                          [0., math.sin(angle_x),  math.cos(angle_x), 0.],
                          [0., 0., 0., 1.]])

    angle_y = angle_y/180.0 * _PI
    Mmatrix_Y = np.array([[ math.cos(angle_y), 0., math.sin(angle_y), 0.],
                          [0., 1., 0., 0.],
                          [-math.sin(angle_y), 0., math.cos(angle_y), 0.],
                          [0., 0., 0., 1.]])

    angle_z = angle_z/180.0 * _PI
    Mmatrix_Z = np.array([[math.cos(angle_z), -math.sin(angle_z), 0., 0.],
                          [math.sin(angle_z),  math.cos(angle_z), 0., 0.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]])

    scale_matrix = np.array([[2.5, 0., 0., 0.],
                             [0., 2.5, 0., 0.],
                             [0., 0., 2.5, 0.],
                             [0., 0., 0. , 1.]], dtype=np.float32)

    translate_matrix = np.array([[1., 0., 0., 0.],
                                 [0., 1., 0., 0.],
                                 [0., 0., 1., 0.],
                                 [0., 0., 0., 1.]], dtype=np.float32)

    rotate_matrix = np.dot(np.dot(Mmatrix_X, Mmatrix_Y), Mmatrix_Z)
    return np.dot(translate_matrix, np.dot(rotate_matrix, scale_matrix))

def get_Vmatrix(camera_pos):
    '''
    @param
        camera_pos --np.array --shape=(3,)
    @return
        Vmatrix --np.array --shape=(4,4)

    视角变换:
      1. 将 camera 平移到 origin
      2. 旋转 camera 看向 -Z 方向 // g
      3. 旋转 camera 的旋转角指向 +Y 方向 // t
      4. 旋转 camera 让 (g×t) 指向 +X 方向
    '''
    # 在本例中, 假设相机已经对齐, 仅考虑平移
    Rmatrix = np.identity(4, dtype=np.float32)
    # 平移矩阵
    x, y, z = camera_pos
    Tmatrix = np.array([[1., 0., 0., -x],
                        [0., 1., 0., -y],
                        [0., 0., 1., -z],
                        [0., 0., 0., 1.]], dtype=np.float32)
    Vmatrix = np.dot(Rmatrix, Tmatrix)
    return Vmatrix

def get_Pmatrix(eye_fov_angle, # 👀 的视角范围 
                aspect_ratio,  # 屏幕宽高比
                z_near,        
                z_far):        # 与近平面、远平面的距离
    # 考虑成角投影
    # 1. 将成角投影转换为正交投影
    perspec2ortho_matrix = np.array([[z_near, 0., 0., 0.],
                                     [0., z_near, 0., 0.],
                                     [0., 0., z_near+z_far, -z_near*z_far],
                                     [0., 0., 1., 0.]], dtype=np.float32)
    # 2. 考虑正交投影: 先平移, 再缩放
    theta = eye_fov_angle/2.0/180.0 * _PI
    top = z_near*math.tan(theta)
    rit = top*aspect_ratio 
    lft = -rit 
    btm = -top 
    # 正交投影平移矩阵
    ortho_Tmatrix = np.array([[1., 0., 0., -(rit+lft)/2],
                              [0., 1., 0., -(top+btm)/2],
                              [0., 0., 1., -(z_near+z_far)/2],
                              [0., 0., 0., 1.]], dtype=np.float32)
    # 正交投影缩放矩阵 -> [-1.0, 1.0]×[-1.0, 1.0]×[-1.0, 1.0]
    ortho_Smatrix = np.array([[2./(rit-lft), 0., 0., 0.],
                              [0., 2./(top-btm), 0., 0.],
                              [0., 0., 2./(z_near-z_far), 0.],
                              [0., 0., 0., 1.]], dtype=np.float32)

    return np.dot(np.dot(ortho_Smatrix, ortho_Tmatrix), perspec2ortho_matrix)

## 这是别人的 P-matrix
# def get_Pmatrix(fov,aspect,near,far):
#     #构建进行透视投影的矩阵
#     t2a=np.tan(fov/2.0)
#     return np.array([[1./(aspect*t2a),0.,0.,0.],[0,1./t2a,0.,0.],[0.,0.,(near+far)/(near-far),2*near*far/(near-far)],[0.,0.,-1.,0.]])

def on_key_press(event):
    key = event.key
    ## 翻转
    if key == "a":
        args.angle_z -= 5 
        plt.close()
    elif key == "d":
        args.angle_z += 5
        plt.close()
    ## 俯仰
    elif key == "up":
        args.angle_x += 5
        plt.close()
    elif key == "down":
        args.angle_x -= 5
        plt.close()
    ## 摇摆
    elif key == "left":
        args.angle_y += 5
        plt.close()
    elif key == "right":
        args.angle_y -= 5
        plt.close()
    elif key == "enter":
        args.quit = True
    print(args.angle_x, args.angle_y, arg.angle_z)

def xyz2xyzw(vertex):
    x, y, z = vertex
    return np.array([x, y, z, 1.], dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW1 CMD.")
    parser.add_argument("--angle_x", type=float, default=0.0)
    parser.add_argument("--angle_y", type=float, default=0.0)
    parser.add_argument("--angle_z", type=float, default=0.0)
    parser.add_argument("--screenshot", type=str, default="screenshot.png")
    parser.add_argument("--obj_path", type=str, default="./models/spot/spot/spot_triangulated_good.obj")

    args = parser.parse_args()
    args.quit = False

    obj_loader = OBJloader()
    obj_loader.load_obj(args.obj_path)

    ## Analysis triangles
    triangles = []
    for mesh in obj_loader.loaded_meshes:
        for i in range(0, len(mesh.indices), 3):
            triangle = Triangle.Triangle()
            for _id in range(3):
                vertex = mesh.vertices[mesh.indices[i+_id]]
                triangle.set_vertex(_id, np.array([vertex.position[0], vertex.position[1], vertex.position[2], 1.], dtype=np.float32))
                triangle.set_normal(_id, np.array([vertex.normal[0], vertex.normal[1], vertex.normal[2]], dtype=np.float32))
                triangle.set_texture_coordinate(_id, np.array([vertex.texture_coordinate[0], vertex.texture_coordinate[1]], dtype=np.float32))
            # print(">>>>>>>>>>>>>>>>>>>>>>>")
            # print(triangle.vertices)
            # print(triangle.colors)
            # print(triangle.normals)
            # print(triangle.textures)
            triangles.append(triangle) 
        #     break
        # break
    
    ## Initialize rasterizer
    rasterizer = Rasterizer.Rasterizer(512+256, 512+256) # 渲染器/屏幕
    # rasterizer = Rasterizer.Rasterizer(700, 700) # 渲染器/屏幕
    # tex_path = args.obj_path.replace(os.path.basename(args.obj_path), "hmap.jpg")
    tex_path = "./models/spot/hmap.jpg"
    rasterizer.set_texture_path(tex_path)
    rasterizer.obj_path = args.obj_path

    _camera_pos = np.array([0, 0, 10], dtype=np.float32)                  # the camera position
    _PI = math.pi

    #### 画图
    while True:
        rasterizer.clear(Rasterizer.COLOR|Rasterizer.DEPTH) # 重置帧缓存和深度缓存
        rasterizer.set_Mmatrix(get_Mmatrix(args.angle_x, args.angle_y, args.angle_z))
        rasterizer.set_Vmatrix(get_Vmatrix(_camera_pos))
        rasterizer.set_Pmatrix(get_Pmatrix(45, 1, 0.1, 50))
    
        if True:#for i in range(0, 12, 2):
            rasterizer.render(triangles)
            screenshot = rasterizer.get_screenshot()

            fig, ax = plt.subplots()
            fig.canvas.mpl_connect("key_press_event", on_key_press)
            plt.imshow(screenshot)
            plt.show()
            # break
        if args.quit:
            break
        # break
    plt.close()









