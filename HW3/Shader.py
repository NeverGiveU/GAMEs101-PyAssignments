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
    return_colors = payload.normals + np.array([1., 1., 1.])
    return (return_colors / 2 * 255.).astype(np.uint8)


def texture_fragment_shader(payload):
    # print(payload.texture_coordinates.min(), payload.texture_coordinates.max())
    texture_colors = payload.texture.get_color(payload.texture_coordinates[:, :, 0],  
                                               payload.texture_coordinates[:, :, 1])

    ka = np.array([[[.0050, .0050, .0050]]], dtype=np.float32)    # (3,)
    kd = texture_colors/255.                               # (h,w,3)
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
    view_directions = _camera_pos-points   # (h,w,3)
    view_directions = view_directions/(np.sqrt(np.power(view_directions, 2).sum(axis=2))+1e-8)[:, :, np.newaxis]

    return_colors = np.zeros_like(colors, dtype=np.float32)
    for light in lights:
        # TODO: For each light source in the env., cacculate what the `ambient`, `diffuse`, and `specular` components are, and
        # accumulate the result on the `return_colors`.
        light_directions = light.position[np.newaxis, np.newaxis, :] - points
        # rr = light.position[:, :, np.newaxis]-points                # (h,w,3) 
        rr = np.sqrt(np.power(light_directions, 2).sum(axis=2))+1e-8 # (h,w), 光源到物体表面某个点的边缘欧几里得距离
        light_directions = light_directions/rr[:, :, np.newaxis]

        component_diffuse = np.zeros(3, dtype=np.float32)
        component_specular= np.zeros(3, dtype=np.float32)
        component_ambient = np.zeros(3, dtype=np.float32)

        h = view_directions+light_directions
        h = h/(np.sqrt(np.power(h, 2).sum(axis=2))+1e-8)[:, :, np.newaxis] # (h,w,3)

        intensity = light.intensity[np.newaxis, np.newaxis, :]/np.power(rr[:, :, np.newaxis], 2) # (h,w,3)

        component_diffuse = kd*intensity*np.clip((normals*light_directions).sum(axis=2)[:, :, np.newaxis], 0., 1e8)
        '''
        .dot
        (h,w,3) .dot (h,w,3) -> (h,w,1)
        '''
        component_specular = ks*intensity*np.power(np.clip((normals*h).sum(axis=2)[:, :, np.newaxis], 0., 1e8), p)
        component_ambient = amb_light_intensity*ka # (1,1,3)

        return_colors = return_colors + component_ambient + component_diffuse + component_specular

    return_colors = np.clip(return_colors, 0., 1.)
    return (return_colors*255.).astype(np.uint8)



