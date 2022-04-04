import numpy as np  
from Texture import Texture 


class FragmentShaderPayload(object):
    def __init__(self, viewspace_pos, colors, normals, texture_coordinates, texture):
        '''
        @param
            viewspace_pos --np.array --shape=(h,w,3)
            colors --np.array --shape=(h,w,3)
            normals --np.array --shape=(h,w,4)
            texture_coordinates --np.array --shape=(h,w,2)
            texture --Texture
        '''
        self.viewspace_pos = viewspace_pos
        self.colors = colors                             
        self.normals = normals 
        self.texture_coordinates = texture_coordinates
        self.texture = texture

        # print(self.viewspace_pos.shape)
        # print(self.colors.shape)
        # print(self.normals.shape)
        # print(self.texture_coordinates.shape)


class Light(object):
    def __init__(self, position, intensity):
        '''
        @param
            position --np.array --shape=(3,)
            intensity --np.array --shape=(3,)
        '''
        self.position = position 
        self.intensity = intensity


## different shaders
'''
@param
    payload --type=FragmentShaderPayload
@return
    return_colors --np.array --shape=(h,w,3)
'''
def normal_fragment_shader(payload):
    '''
    使用 normalized 的法向量 
    --shape=(h,w,3) --range=[-1., 1.]

    替代 R/G/B 的颜色值
    '''
    # return_colors = payload.normals
    # return_colors = np.clip(return_colors, 0., 1.)
    # return (return_colors*255.).astype(np.uint8)
    return_colors = payload.normals + np.array([1., 1., 1.]) # range=[-1., 1.] -> [0., 2.]
    return (return_colors / 2 * 255.).astype(np.uint8)       # range=[0., 2.] -> [0., 1.] -> [0., 255.]


def texture_fragment_shader(payload):
    texture_colors = payload.texture.get_color(payload.texture_coordinates[:, :, 0],  
                                               payload.texture_coordinates[:, :, 1])

    ka = np.array([[[.0050, .0050, .0050]]], dtype=np.float32) # (3,)
    kd = texture_colors/255.                                   # (h,w,3)
    ks = np.array([[[.7937, .7937, .7937]]], dtype=np.float32) # (3,)

    # lights
    l1 = Light(
        position=np.array([20., 20., 20.], dtype=np.float32),
        intensity=np.array([500., 500., 500.], dtype=np.float32)
    )
    l2 = Light(
        position=np.array([-20., 20., 0.], dtype=np.float32),
        intensity=np.array([500., 500., 500.], dtype=np.float32)
    )
    lights = [l1, l2]

    # 
    amb_light_intensity = np.array([[[10., 10., 10.]]], dtype=np.float32) # 环境光强度
    _camera_pos = np.array([0, 0, 10], dtype=np.float32)                  # the camera position

    p = 150.
    colors = texture_colors
    points = payload.viewspace_pos
    normals = payload.normals # (h,w,3)
    '''
    计算向量 v: 从着色点指向相机中心
    i.e., 用相机坐标减去着色点坐标
    '''
    view_directions = _camera_pos-points   # (h,w,3)
    view_directions = view_directions/(np.sqrt(np.power(view_directions, 2).sum(axis=2))+1e-8)[:, :, np.newaxis]

    return_colors = np.zeros_like(colors, dtype=np.float32)
    for light in lights:
        # TODO: For each light source in the env., cacculate what the `ambient`, `diffuse`, and `specular` components are, and
        # accumulate the result on the `return_colors`.
        '''
        计算向量 l: 从着色点指向光源
        i.e., 用光源坐标减去着色点坐标
        '''
        light_directions = light.position[np.newaxis, np.newaxis, :] - points
        # rr = light.position[:, :, np.newaxis]-points                # (h,w,3) 
        rr = np.sqrt(np.power(light_directions, 2).sum(axis=2))+1e-8 # (h,w), 光源到物体表面某个点的边缘欧几里得距离
        light_directions = light_directions/rr[:, :, np.newaxis]

        ## 计算高光项时光线方向与相机方向之间的角平分线
        h = view_directions+light_directions
        h = h/(np.sqrt(np.power(h, 2).sum(axis=2))+1e-8)[:, :, np.newaxis] # (h,w,3)

        intensity = light.intensity[np.newaxis, np.newaxis, :]/np.power(rr[:, :, np.newaxis], 2) # (h,w,3)
        # 计算漫反射和高光项时, 考虑光在着色点处的衰减

        component_diffuse = kd*intensity*np.clip((normals*light_directions).sum(axis=2)[:, :, np.newaxis], 0., 1e8)
        #                                        (normals*light_directions).sum(axis=2) ## 求 cosθ, 并且设置角度超过 180° (即: cos θ <= 0) 的点为 0
        #                   kd 是反射系数, 一般认为是什么颜色的物体就反射回什么颜色的光
        #                      intensity 是 R/G/B 
        # 计算漫反射的光强
        '''
        .dot
        (h,w,3) .dot (h,w,3) -> (h,w,1)
        '''
        component_specular = ks*intensity*np.power(np.clip((normals*h).sum(axis=2)[:, :, np.newaxis], 0., 1e8), p)
        #                    ks 是高光系数
        #                       intensity 是光源的强度
        #                                 (normals * n).sum(axis=2)  ## 求 cos α, 并且设置角度超过 180° (即: cos α <=0) 的点为 0.
        # 计算高光项

        component_ambient = amb_light_intensity*ka # (1,1,3)

        return_colors = return_colors + component_ambient + component_diffuse + component_specular

    return_colors = np.clip(return_colors, 0., 1.)
    return (return_colors*255.).astype(np.uint8)


def phong_fragment_shader(payload):
    '''
    Phong shading 的思想是: 对三角形碎片的顶点的法向量作插值
    : Interpolate normal vectors across each triangle
    : Compute full shading model at each pixel

    ❗❗ 巧了, 
    我们传入 payload 的 normals 恰好是经过每个顶点的法向量作重心插值得到的, 
    因此, 除了漫反射的 kd 不同外, 其他的与 texture_fragment_shader 是一样的
    '''
    ka = np.array([[[.0050, .0050, .0050]]], dtype=np.float32) # (3,)
    kd = payload.colors/255.                                   # (h,w,3)
    ks = np.array([[[.7937, .7937, .7937]]], dtype=np.float32) # (3,)

    # lights
    l1 = Light(
        position=np.array([20., 20., 20.], dtype=np.float32),
        intensity=np.array([500., 500., 500.], dtype=np.float32)
    )
    l2 = Light(
        position=np.array([-20., 20., 0.], dtype=np.float32),
        intensity=np.array([500., 500., 500.], dtype=np.float32)
    )
    lights = [l1, l2]

    amb_light_intensity = np.array([[[10., 10., 10.]]], dtype=np.float32) # 环境光强度
    _camera_pos = np.array([0, 0, 10], dtype=np.float32)                  # the camera position

    p = 150.

    return_colors = np.zeros_like(payload.colors, dtype=np.float32)
    points = payload.viewspace_pos
    normals = payload.normals # (h,w,3)
    view_directions = _camera_pos-points   # (h,w,3)
    view_directions = view_directions/(np.sqrt(np.power(view_directions, 2).sum(axis=2))+1e-8)[:, :, np.newaxis]

    for light in lights:
        light_directions = light.position[np.newaxis, np.newaxis, :] - points
        rr = np.sqrt(np.power(light_directions, 2).sum(axis=2))+1e-8 # (h,w), 光源到物体表面某个点的边缘欧几里得距离
        light_directions = light_directions/rr[:, :, np.newaxis]

        h = view_directions+light_directions
        h = h/(np.sqrt(np.power(h, 2).sum(axis=2))+1e-8)[:, :, np.newaxis] # (h,w,3)

        intensity = light.intensity[np.newaxis, np.newaxis, :]/np.power(rr[:, :, np.newaxis], 2) # (h,w,3)
        component_diffuse = kd*intensity*np.clip((normals*light_directions).sum(axis=2)[:, :, np.newaxis], 0., 1e8)
        component_specular = ks*intensity*np.power(np.clip((normals*h).sum(axis=2)[:, :, np.newaxis], 0., 1e8), p)
        component_ambient = amb_light_intensity*ka # (1,1,3)

        return_colors = return_colors + component_ambient + component_diffuse + component_specular
    return_colors = np.clip(return_colors, 0., 1.)
    return (return_colors*255.).astype(np.uint8)


## 探究 texture 在纹理映射中的其他应用
def bump_fragment_shader(payload):
    '''
    bump/normal mapping
    : 纹理图的每个点的 r/g/b 记录的不是颜色, 而是该点在三个维度对应的 shift offsets
    
    (x, y, z) -> (x+dx, y+dy, z+dz)

    那么对应的法向量就也需要发生变化
    (nx,ny,nz) -> ?
    '''
    ka = np.array([[[.0050, .0050, .0050]]], dtype=np.float32) # (3,)
    kd = payload.colors/255.                                   # (h,w,3)
    ks = np.array([[[.7937, .7937, .7937]]], dtype=np.float32) # (3,)

    # lights
    l1 = Light(
        position=np.array([20., 20., 20.], dtype=np.float32),
        intensity=np.array([500., 500., 500.], dtype=np.float32)
    )
    l2 = Light(
        position=np.array([-20., 20., 0.], dtype=np.float32),
        intensity=np.array([500., 500., 500.], dtype=np.float32)
    )
    lights = [l1, l2]

    amb_light_intensity = np.array([[[10., 10., 10.]]], dtype=np.float32) # 环境光强度
    _camera_pos = np.array([0, 0, 10], dtype=np.float32)                  # the camera position

    p = 150.
    
    ## data for shadering
    # return_colors = np.zeros_like(payload.colors, dtype=np.float32)
    # points = payload.viewspace_pos
    normals = payload.normals # (h,w,3)

    ## 全局变换的矩阵
    kh = 0.2 
    kn = 0.1
    nx = normals[:, :, 0:1]
    ny = normals[:, :, 1:2]
    nz = normals[:, :, 2:3]

    nx2_nz2_sqrt = np.sqrt(np.power(nx, 2)+np.power(nz, 2)) # 法向量在 xOz 平面上的投影大小
    t = np.concatenate([
        nx*ny/nx2_nz2_sqrt,
        nx2_nz2_sqrt,       # i think it should be `-nx2_nz2_sqrt`
        nz*ny/nx2_nz2_sqrt
    ], axis=2)
    b = np.cross(normals, t) # (h,w,3)

    tbn = np.concatenate([
        t[:, :, :, np.newaxis], 
        b[:, :, :, np.newaxis],
        normals[:, :, :, np.newaxis]
    ], axis=3)               # (h,w,3,three)
    '''
    对于每个点, tbn[i,j] 是一个矩阵: 
       [[tx, bx, nx],
        [ty, by, ny],
        [tz, bz, nz]]

    ln (local normal) 则是一个列向量: 
       [[lx],
        [ly],
        [lz]]
    # 作乘法 (实际上是实现了点乘)
       [[tx*lx, bx*lx, nx*lx], -> [[tx*lx+bx*lx+nx*lx],
        [ty*ly, by*ly, ny*ly], ->  [ty*ly+by*ly+ny*ly],
        [tz*lz, bz*lz, nz*lz]] ->  [tz*lz+bz*lz+nz*lz]]

    tx = nx*ny/sqrt(nx^2+nz^2)
    ty = sqrt(nx^2+nz^2)
    tz = nz*ny/sqrt(nx^2+nz^2)
    b  = [nx,ny,nz]×[tx,ty,tz] = [ny*tz-ty*nz, nz*tx-nx*tz, nx*ty-ny*tx]
                               = [ny^2nz/sqrt-nz*sqrt,
                                  nx*ny*nz/sqrt-nx*ny*nz/sqrt=0.,
                                  nx*sqrt-nx*ny^2/sqrt]
        tx*lx+bx*lx+nx*lx = lx*nx*ny/sqrt + lx*(ny^2nz/sqrt-nz*sqrt) + lx*nx
                          = lx*(nx*ny/sqrt + ny^2nz/sqrt-nz*sqrt + nx)
                          = lx*[(nx*ny+ny*ny*nz-nz*(nx*nx+nz*nz))/sqrt + nx]
    '''

    ## 于是计算新的局部法向量如下: 
    W = payload.texture.width
    H = payload.texture.height

    u = payload.texture_coordinates[:, :, 0]
    v = payload.texture_coordinates[:, :, 1]

    huv = payload.texture.get_color(u, v)
    huv_norm = np.sqrt(np.power(huv, 2).sum(axis=2))[:, :, np.newaxis]
    huv = huv/huv_norm

    u_shift = u+1./W
    v_shift = v+1./H 
    huv_ushift_v = payload.texture.get_color(u_shift, v)
    huv_ushift_v_norm = np.sqrt(np.power(huv_ushift_v, 2).sum(axis=2))[:, :, np.newaxis]
    huv_u_vshift = payload.texture.get_color(u, v_shift)
    huv_u_vshift_norm = np.sqrt(np.power(huv_u_vshift, 2).sum(axis=2))[:, :, np.newaxis]

    dU = kh * kn * (huv_ushift_v_norm-huv_norm)
    dV = kh * kn * (huv_u_vshift_norm-huv_norm)
    '''
    *_norm 是欧几里得距离, 因为此时纹理图表示的是 "三角形平面内, 每个点在三个维度的 '海拔'", 
    那么 shift*_norm - *_norm 就表示: 
        
    想象一下, 我们假设是一幅等高线图, 此时仅考虑一维的 offset
    那么, shift*_norm - *_norm 就得到了在这个维度的高度差, 
        (dx, 0) -> (dx, dh)
    根据法向量点积为 0 可以反推出:
        normal' = (-dh, dx)
        (-dh, dx).dot((dx, dh)) = -dxdh+dxdh = 0.

    当然, 这个新的法向量是在局部空间推算的

    那么在二维的 uv-map 也是类似的:
        local_normal' = (-du, -dv, dx=1.)
    '''
    ln = np.concatenate([
        -dU[:, :, :, np.newaxis],
        -dV[:, :, :, np.newaxis],
        np.ones_like(dU)[:, :, :, np.newaxis]
    ], axis=3) # (h,w,1,three) # 新的法向量

    return_colors = (tbn*ln).sum(axis=3)
    '''
    对于每个点, tbn[i,j] 是一个矩阵: 
       [[tx, bx, nx],
        [ty, by, ny],
        [tz, bz, nz]]

    ln (local normal) 则是一个列向量: 
       [[lx],
        [ly],
        [lz]]
    # 作乘法 (实际上是实现了点乘)
       [[tx*lx, bx*lx, nx*lx], -> [[tx*lx+bx*lx+nx*lx],
        [ty*ly, by*ly, ny*ly], ->  [ty*ly+by*ly+ny*ly],
        [tz*lz, bz*lz, nz*lz]] ->  [tz*lz+bz*lz+nz*lz]]
    '''

    ## 归一化后, 直接拿结果作为颜色, 类似于 normal_fragment_shader
    return_colors = return_colors/np.sqrt(np.power(return_colors, 2).sum(axis=2))[:, :, np.newaxis]
    return_colors = np.clip(return_colors, 0., 1.)

    return (return_colors*255.).astype(np.uint8)

    
def displacement_fragment_shader(payload):
    ka = np.array([[[.0050, .0050, .0050]]], dtype=np.float32) # (3,)
    kd = payload.colors/255.                                   # (h,w,3)
    ks = np.array([[[.7937, .7937, .7937]]], dtype=np.float32) # (3,)

    # lights
    l1 = Light(
        position=np.array([20., 20., 20.], dtype=np.float32),
        intensity=np.array([500., 500., 500.], dtype=np.float32)
    )
    l2 = Light(
        position=np.array([-20., 20., 0.], dtype=np.float32),
        intensity=np.array([500., 500., 500.], dtype=np.float32)
    )
    lights = [l1, l2]

    amb_light_intensity = np.array([[[10., 10., 10.]]], dtype=np.float32) # 环境光强度
    _camera_pos = np.array([0, 0, 10], dtype=np.float32)                  # the camera position
    p = 150.
    
    ## 
    points = payload.viewspace_pos
    normals = payload.normals # (h,w,3)
    kh = 0.2 
    kn = 0.1 # bump mapping 中的系数
    nx = normals[:, :, 0:1]
    ny = normals[:, :, 1:2]
    nz = normals[:, :, 2:3]
    nx2_nz2_sqrt = np.sqrt(np.power(nx, 2)+np.power(nz, 2))
    t = np.concatenate([
        nx*ny/nx2_nz2_sqrt,
        nx2_nz2_sqrt,     # i think it should be `-nx2_nz2_sqrt`
        nz*ny/nx2_nz2_sqrt
    ], axis=2)
    b = np.cross(normals, t) # (h,w,3)
    t = t/np.sqrt(np.power(t, 2).sum(axis=2)+1e-8)[:, :, np.newaxis] # (h,w,3)
    b = b/np.sqrt(np.power(b, 2).sum(axis=2)+1e-8)[:, :, np.newaxis] # (h,w,3)

    tbn = np.concatenate([
        t[:, :, :, np.newaxis], 
        b[:, :, :, np.newaxis],
        normals[:, :, :, np.newaxis]
    ], axis=3)               # (h,w,3,three)

    # 
    W = payload.texture.width
    H = payload.texture.height

    u = payload.texture_coordinates[:, :, 0]
    v = payload.texture_coordinates[:, :, 1]

    huv = payload.texture.get_color(u, v)
    huv_norm = np.sqrt(np.power(huv, 2).sum(axis=2))[:, :, np.newaxis]
    huv = huv/huv_norm

    u_shift = u+1./W
    v_shift = v+1./H 
    huv_ushift_v = payload.texture.get_color(u_shift, v)
    huv_ushift_v_norm = np.sqrt(np.power(huv_ushift_v, 2).sum(axis=2))[:, :, np.newaxis]
    huv_u_vshift = payload.texture.get_color(u, v_shift)
    huv_u_vshift_norm = np.sqrt(np.power(huv_u_vshift, 2).sum(axis=2))[:, :, np.newaxis]

    dU = kh * kn * (huv_ushift_v_norm-huv_norm)
    dV = kh * kn * (huv_u_vshift_norm-huv_norm)
    
    ln = np.concatenate([
        -dU[:, :, :, np.newaxis],
        -dV[:, :, :, np.newaxis],
        np.ones_like(dU)[:, :, :, np.newaxis]
    ], axis=3) # (h,w,1,three)

    n = (tbn*ln).sum(axis=3)
    n = n/np.sqrt(np.power(n, 2).sum(axis=2))[:, :, np.newaxis]
    
    points = points + n*huv_norm*kn ## TODO: 直接改变点的位置 -> 直接改变了对象的形状

    return_colors = np.zeros_like(kd, dtype=np.float32)
    
    view_directions = _camera_pos-points   # (h,w,3)
    view_directions = view_directions/(np.sqrt(np.power(view_directions, 2).sum(axis=2))+1e-8)[:, :, np.newaxis]
    for light in lights:
        light_directions = light.position[np.newaxis, np.newaxis, :] - points
        rr = np.sqrt(np.power(light_directions, 2).sum(axis=2))+1e-8 # (h,w), 光源到物体表面某个点的边缘欧几里得距离
        light_directions = light_directions/rr[:, :, np.newaxis]

        h = view_directions+light_directions
        h = h/(np.sqrt(np.power(h, 2).sum(axis=2))+1e-8)[:, :, np.newaxis] # (h,w,3)

        intensity = light.intensity[np.newaxis, np.newaxis, :]/np.power(rr[:, :, np.newaxis], 2) # (h,w,3)
        component_diffuse = kd*intensity*np.clip((normals*light_directions).sum(axis=2)[:, :, np.newaxis], 0., 1e8)
        component_specular = ks*intensity*np.power(np.clip((normals*h).sum(axis=2)[:, :, np.newaxis], 0., 1e8), p)
        component_ambient = amb_light_intensity*ka # (1,1,3)

        return_colors = return_colors + component_ambient + component_diffuse + component_specular
    return_colors = np.clip(return_colors, 0., 1.)
    return (return_colors*255.).astype(np.uint8)