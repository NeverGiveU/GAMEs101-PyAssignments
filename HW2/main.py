import numpy as np 
import math 
import Triangle
import Rasterizer
import argparse
import matplotlib.pyplot as plt 


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

    return np.dot(np.dot(Mmatrix_X, Mmatrix_Y), Mmatrix_Z)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW1 CMD.")
    parser.add_argument("--angle_x", type=float, default=0.0)
    parser.add_argument("--angle_y", type=float, default=0.0)
    parser.add_argument("--angle_z", type=float, default=0.0)
    parser.add_argument("--screenshot", type=str, default="screenshot.png")

    args = parser.parse_args()
    args.quit = False

    #### Global variables
    _PI = math.pi                                      # pi
    _camera_pos = np.array([0, 0, 5])                  # the camera position
    vertices = np.array([[2., 0., -2.],
                         [0., 2., -2.],
                         [-2., 0., -2.], # primitivity 1
                         [3.5, -1., -5.],
                         [2.5, 1.5, -5.],
                         [-1., 0.5, -5.] # primitivity 2
                         ])                            # vertices of the triangle
    indices = np.array([[0, 1, 2],
                        [3, 4, 5]], dtype=np.uint8)    # indices, each 3d-tuple represents a triangle primitivity
    colors = np.array([[217., 238., 185.],
                       [217., 238., 185.],
                       [217., 238., 185.],
                       [185., 217., 238.],
                       [185., 217., 238.],
                       [185., 217., 238.]], dtype=np.float32)

    key = 0
    frame_count = 0                                    # 帧计数器
    
    #### 初始化
    rasterizer = Rasterizer.Rasterizer(512+256, 512+256) # 渲染器/屏幕
    v_id = rasterizer.load_vertices(vertices)
    i_id = rasterizer.load_indices(indices)
    c_id = rasterizer.load_colors(colors)

    #### 画图
    while True:
        rasterizer.clear(Rasterizer.COLOR|Rasterizer.DEPTH) # 重置帧缓存和深度缓存
        rasterizer.set_Mmatrix(get_Mmatrix(args.angle_x, args.angle_y, args.angle_z))
        rasterizer.set_Vmatrix(get_Vmatrix(_camera_pos))
        rasterizer.set_Pmatrix(get_Pmatrix(45, 1, 0.1, 50))

        rasterizer.rasterize(v_id, i_id, c_id, ptype="Triangle")
        screenshot = rasterizer.get_screenshot()

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("key_press_event", on_key_press)
        plt.imshow(screenshot)
        plt.show()

        if args.quit:
            break
        # break
    plt.close()
