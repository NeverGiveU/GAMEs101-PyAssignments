from .Triangle import Triangle
import numpy as np 


def mesh_subdivision_loop(triangles):
    cache = {} ## 存储每个点出现对应的id
    for i, triangle in enumerate(triangles):
        for v in triangle.vertices:
            v = str(v)
            if v in cache:
                cache[v].append(i)
            else:
                cache[v] = [i]
    for k in cache:
        cache[k] = set(cache[k])

    new_triangles = []
    ## 将每个三角形分为4个小三角形
    for i, triangle in enumerate(triangles):
        v1, v2, v3 = triangle.vertices
        t1, t2, t3 = triangle.textures
        n1, n2, n3 = triangle.normals

        sv1, sv2, sv3 = str(v1), str(v2), str(v3)
        # split
        v12 = (v1+v2)/2
        v23 = (v2+v3)/2
        v31 = (v3+v1)/2
        # update
        # v12
        i1, i2 = (cache[sv1]).intersection(cache[sv2])
        A = v1
        B = v2 
        C = v3 
        j = i1 if i == i2 else i2
        triangle = triangles[j]
        for v, t, n in zip(triangle.vertices, triangle.textures, triangle.normals):
            sv = str(v)
            if sv == sv1 or sv == sv2:
                continue
            else:
                D = v 
                break
        v12 = .375*(A+B)+.125*(C+D)
        t12 = .375*(t1+t2)+.125*(t3+t)
        n12 = .375*(n1+n2)+.125*(n3+n)
        # v23
        i2, i3 = (cache[sv2]).intersection(cache[sv3])
        A = v2
        B = v3 
        C = v1 
        j = i2 if i == i3 else i3
        triangle = triangles[j]
        for v, t, n in zip(triangle.vertices, triangle.textures, triangle.normals):
            sv = str(v)
            if sv == sv2 or sv == sv3:
                continue
            else:
                D = v 
                break
        v23 = .375*(A+B)+.125*(C+D)
        t23 = .375*(t2+t3)+.125*(t1+t)
        n23 = .375*(n2+n3)+.125*(n1+n)
        # v31
        i3, i1 = (cache[sv3]).intersection(cache[sv1])
        A = v3
        B = v1 
        C = v2 
        j = i3 if i == i1 else i1
        triangle = triangles[j]
        for v, t, n in zip(triangle.vertices, triangle.textures, triangle.normals):
            sv = str(v)
            if sv == sv3 or sv == sv1:
                continue
            else:
                D = v 
                break
        v31 = .375*(A+B)+.125*(C+D)
        t31 = .375*(t3+t1)+.125*(t2+t)
        n31 = .375*(n3+n1)+.125*(n2+n)

        # v1
        cache_tmp = {}
        v_tmp = []
        t_tmp = []
        n_tmp = []
        for j in cache[sv1]:
            triangle = triangles[j]
            for v, t, n in zip(triangle.vertices, triangle.textures, triangle.normals):
                sv = str(v)
                if sv != sv1 and sv not in cache_tmp:
                    cache_tmp[sv] = True 
                    v_tmp.append(v)
                    t_tmp.append(t)
                    n_tmp.append(n)
        n = len(v_tmp)
        u = 3/16 if n == 3 else 3/(8*n)
        alpha = 1-n*u 
        beta = u 
        v1_ = v1*alpha + beta*np.array(v_tmp).sum(axis=0)
        t1_ = t1*alpha + beta*np.array(t_tmp).sum(axis=0)
        n1_ = n1*alpha + beta*np.array(n_tmp).sum(axis=0)
        # v2
        cache_tmp = {}
        v_tmp = []
        t_tmp = []
        n_tmp = []
        for j in cache[sv2]:
            triangle = triangles[j]
            for v, t, n in zip(triangle.vertices, triangle.textures, triangle.normals):
                sv = str(v)
                if sv != sv2 and sv not in cache_tmp:
                    cache_tmp[sv] = True 
                    v_tmp.append(v)
                    t_tmp.append(t)
                    n_tmp.append(n)
        n = len(v_tmp)
        u = 3/16 if n == 3 else 3/(8*n)
        alpha = 1-n*u 
        beta = u 
        v2_ = v2*alpha + beta*np.array(v_tmp).sum(axis=0)
        t2_ = t2*alpha + beta*np.array(t_tmp).sum(axis=0)
        n2_ = n2*alpha + beta*np.array(n_tmp).sum(axis=0)
        # v3
        cache_tmp = {}
        v_tmp = []
        t_tmp = []
        n_tmp = []
        for j in cache[sv3]:
            triangle = triangles[j]
            for v, t, n in zip(triangle.vertices, triangle.textures, triangle.normals):
                sv = str(v)
                if sv != sv3 and sv not in cache_tmp:
                    cache_tmp[sv] = True 
                    v_tmp.append(v)
                    t_tmp.append(t)
                    n_tmp.append(n)
        n = len(v_tmp)
        u = 3/16 if n == 3 else 3/(8*n)
        alpha = 1-n*u 
        beta = u 
        v3_ = v3*alpha + beta*np.array(v_tmp).sum(axis=0)
        t3_ = t3*alpha + beta*np.array(t_tmp).sum(axis=0)
        n3_ = n3*alpha + beta*np.array(n_tmp).sum(axis=0)

        ## append
        tri_1_12_31 = Triangle()
        tri_1_12_31.set_vertex            (0, v1_)
        tri_1_12_31.set_vertex            (1, v12)
        tri_1_12_31.set_vertex            (2, v31)
        tri_1_12_31.set_texture_coordinate(0, t1_)
        tri_1_12_31.set_texture_coordinate(1, t12)
        tri_1_12_31.set_texture_coordinate(2, t31)
        tri_1_12_31.set_normal            (0, n1_)
        tri_1_12_31.set_normal            (1, n12)
        tri_1_12_31.set_normal            (2, n31)
        tri_2_23_12 = Triangle()
        tri_2_23_12.set_vertex            (0, v2_)
        tri_2_23_12.set_vertex            (1, v23)
        tri_2_23_12.set_vertex            (2, v12)
        tri_2_23_12.set_texture_coordinate(0, t2_)
        tri_2_23_12.set_texture_coordinate(1, t23)
        tri_2_23_12.set_texture_coordinate(2, t12)
        tri_2_23_12.set_normal            (0, n2_)
        tri_2_23_12.set_normal            (1, n23)
        tri_2_23_12.set_normal            (2, n12)
        tri_3_31_23 = Triangle()
        tri_3_31_23.set_vertex            (0, v3_)
        tri_3_31_23.set_vertex            (1, v31)
        tri_3_31_23.set_vertex            (2, v23)
        tri_3_31_23.set_texture_coordinate(0, t3_)
        tri_3_31_23.set_texture_coordinate(1, t31)
        tri_3_31_23.set_texture_coordinate(2, t23)
        tri_3_31_23.set_normal            (0, n3_)
        tri_3_31_23.set_normal            (1, n31)
        tri_3_31_23.set_normal            (2, n23)
        tri_12_23_31 = Triangle()
        tri_12_23_31.set_vertex            (0, v12)
        tri_12_23_31.set_vertex            (1, v23)
        tri_12_23_31.set_vertex            (2, v31)
        tri_12_23_31.set_texture_coordinate(0, t12)
        tri_12_23_31.set_texture_coordinate(1, t23)
        tri_12_23_31.set_texture_coordinate(2, t31)
        tri_12_23_31.set_normal            (0, n12)
        tri_12_23_31.set_normal            (1, n23)
        tri_12_23_31.set_normal            (2, n31)

        new_triangles.append(tri_1_12_31)
        new_triangles.append(tri_2_23_12)
        new_triangles.append(tri_3_31_23)
        new_triangles.append(tri_12_23_31)
        #break
    return new_triangles
        




