import numpy as np
import cv2


def normalize(x):
    norm2=x[0]**2+x[1]**2+x[2]**2
    if(norm2>0):
        return x/np.sqrt(norm2)
    else:
        return x

def triangle_intersect(v0,v1,v2,o,d):
    #求v0,v1,v2对应的三角形和o+td形成的光线的交点,返回交点对应的t和重心坐标
    # print(v0, v1, v2)
    e1=(v1-v0).astype(np.float32)
    e2=(v2-v0).astype(np.float32)
    s=o-(v0).astype(np.float32)
    # print(o, v0, s)
    s1=np.cross(d,e2)
    s2=np.cross(s,e1)
    # print(s, e1)
    s1e1=np.dot(s1,e1)
    # print(s2, e2, s1e1)
    t=np.dot(s2,e2)/s1e1
    u=np.dot(s1,s)/s1e1
    v=np.dot(s2,d)/s1e1
    # print(">", v0, v1, v2, o, d, t, u, v)
    if(t>0 and v>=0 and v<=1 and u>=0 and u<=1):
        return t,u,v
    return None,None,None


class Object:
    #一个描述渲染物体的类,有材质,以及渲染需要的一些参数(kd,ks),还有折射率ior
    #默认材质为光滑散射
    material="diffuse and glossy"
    ior=1.3
    kd=0.8
    ks=0.2
    diffuse_color=np.array([2.]*3)
    specular_exponent=25
    def __init__(self):
        pass
    def intersect(self,o,d):
        #对由o+td定义的一条光线,计算它和当前物体的第一个交点,返回较近的t和重心坐标
        pass
    def get_surface_properties(self,p,d,index,uv):
        #求相交点p在表面上的属性,这里包括法向量和坐标(只用于网格,实现格子的效果)
        pass
    def eval_diffusecolor(self,st):
        #根据坐标计算散射颜色
        return self.diffuse_color

class Sphere(Object):
    #一个由中心点,半径,表面材质表示的球类
    def __init__(self,center,radius):
        self.center = center
        self.radius = radius
    def intersect(self,o,d):
        #对由o+td定义的一条光线,计算它和当前物体的第一个交点,返回较近的t
        l=o-self.center
        a=np.linalg.norm(d)**2
        b=2*l.dot(d)
        c=l.dot(l)-self.radius**2
        #进行二次根的求解并返回
        return self.quadratic_solve(a,b,c),None,None
    def quadratic_solve(self,a,b,c):
        #二次根的求解,只要解中非负最小的那个
        delta=b**2-4*a*c
        if(delta<0):
            #只有两个虚根
            return None
        elif(delta==0):
            #只有一个实根$\cfrac{-b}{2a}$
            r = -0.5 * b / a
        else:
            #两个实根$\cfrac{-b\pm\sqrt{b^2-4ac}}{2a}$
            q = -0.5 * (b + np.sqrt(delta)) if b>0 else  -0.5 * (b - np.sqrt(delta))
            x0 = q / a
            x1 = c / q
            r=min(x0,x1)
            if(r<0):
                r=max(x0,x1)
        if(r<0):
            return None
        return r
    def get_surface_properties(self,p,d,index,uv):
        #求相交点p在表面上的属性,这里包括法向量
        return normalize(p-self.center),None

class MeshTriangle(Object):
    #网格状三角形型的物体,v是顶点集合,vindex是三三一组组成三角形的顶点序号集合,st是平面的格子的坐标
    def __init__(self,v,vindex,st):
        self.v=v
        self.vindex=vindex
        self.st=st
    def intersect(self,o,d):
        # print(">>", o, d)
        #对由o+td定义的一条光线,计算它和当前物体的第一个交点,返回较近的t,重心坐标和对应三角形的vindex里的序号
        l=len(vindex)
        tnear=np.inf
        uv=[]
        index=-1
        for k in range(0,l,3):
            #每次取三个顶点作为三角形,进行光线和三角形相交的计算
            t,u,v=triangle_intersect(self.v[self.vindex[k]],self.v[self.vindex[k+1]],self.v[self.vindex[k+2]],o,d)
            # print("<<<<", t, tnear, u, v)
            if(t and t<tnear):
                tnear=t
                uv=[u,v]
                index=k
        # print(">>", tnear, uv, index)
        if index!=-1:
            return tnear,uv,index
        else:
            return None,None,None
    def get_surface_properties(self,p,d,index,uv):
        #求相交点p在表面上的属性,这里包括法向量和坐标
        #先取出序号对应三角形的三个顶点
        v0=self.v[self.vindex[index]]
        v1=self.v[self.vindex[index+1]]
        v2=self.v[self.vindex[index+2]]
        #然后通过两边叉乘得到法向量
        e0 = normalize(v1 - v0)
        e1 = normalize(v2 - v1)
        normal = normalize(np.cross(e0, e1))
        #插值计算了st坐标
        st0 = self.st[self.vindex[index]]
        st1 = self.st[self.vindex[index + 1]]
        st2 = self.st[self.vindex[index + 2]]
        st = st0 * (1 - uv[0] - uv[1]) + st1 * uv[0] + st2 * uv[1]
        return normal,st
    def eval_diffusecolor(self,st):
        # print(st)
        #根据st坐标改变散射颜色,实现格子的变色
        scale=5.
        pattern=(st[0]*scale%1.>0.5)^(st[1]*scale%1.>0.5)
        return np.array([0.937, 0.937, 0.231]) if pattern else np.array([0.815, 0.235, 0.031])


def deg2rad(deg):
    #角度制到弧度制的转换
    return deg/180*np.pi
def clamp(small,large,number):
    #将number规格化,小于small等于small,大于large等于large
    return min(large,max(small,number))
def fresnel(i,n,ior):
    #解菲涅尔方程
    cosi = clamp(-1, 1, np.dot(i, n));
    etai = 1.
    etat = ior
    if (cosi > 0):
        etai,etat=etat,etai
    sint = etai / etat * np.sqrt(max(0., 1 - cosi * cosi))
    if (sint >= 1):
        return 1
    else:
        cost = np.sqrt(max(0., 1 - sint * sint))
        cosi = np.fabs(cosi)
        Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))
        Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost))
        return (Rs * Rs + Rp * Rp) / 2
def reflect(i,n):
    #计算反射的方向,n为法向量,i为入射光方向
    return i-2*np.dot(i,n)*n
def refract(i,normal,ior):
    print(">>in>>", i, normal, ior)
    #计算折射的方向,normal为法向量,i为入射光方向,ior为折射率
    cosi=clamp(-1,1,np.dot(i,normal))
    etai = 1
    etat = ior
    n=normal
    if (cosi < 0):
        cosi = -cosi
    else:
        etai,etat=etat,etai
        n= -normal
    eta = etai / etat
    k = 1 - eta * eta * (1 - cosi * cosi)
    print(cosi, eta, k)
    return np.array([0]*3) if k < 0 else eta * i + (eta * cosi - np.sqrt(k)) * n

def trace(o,d,obj):
    #计算o+td和obj中物体的第一个交点(最小t)
    t=np.inf
    uv=np.zeros(2)
    index=-1
    hit_obj=None
    for i in obj:
        #遍历对象,进行相交的计算
        tnear,uv_temp,index_temp=i.intersect(o,d)
        # print(tnear, index_temp, uv_temp)
        if(tnear and tnear<t):
            t=tnear
            uv=uv_temp
            index=index_temp
            hit_obj=i
    if index!=-1:
        return t,uv,index,hit_obj
    else:
        return None,None,None,None

def cast_ray(o,d,obj,lights,depth):
    #进行光线追踪的计算
    if(depth>5):
        #最多经过5次方向变换
        return np.zeros(3)#, False
    #交点的默认颜色
    hit_color=np.array([0.235294, 0.67451, 0.843137])
    tnear,uv,index,hit_obj=trace(o,d,obj)
    epsilon= 0.00001
    # print("|||", tnear, uv)
    f = False
    if(tnear!=None):
        hit_point=o + d * tnear
        n,st=hit_obj.get_surface_properties(hit_point,d,index,uv)
        if(hit_obj.material=="reflection and refraction"):
            #反射光方向
            reflection_direction = normalize(reflect(d, n))
            #折射光方向
            refraction_direction = normalize(refract(d, n, hit_obj.ior))
            print(reflection_direction, refraction_direction)
            #反射光起点
            reflection_ray_orig = hit_point - n * epsilon if(np.dot(reflection_direction, n) < 0) else\
                                hit_point + n * epsilon
            #折射光起点
            refraction_ray_orig = hit_point - n * epsilon if(np.dot(refraction_direction, n) < 0) else\
                                            hit_point + n * epsilon
            #进行反射光和折射光的光线追踪
            reflection_color = cast_ray(reflection_ray_orig, reflection_direction, obj, lights, depth + 1)
            refraction_color = cast_ray(refraction_ray_orig, refraction_direction, obj, lights, depth + 1)
            kr = fresnel(d, n, hit_obj.ior)
            #进行反射光和折射光的颜色合成
            hit_color = reflection_color * kr + refraction_color * (1 - kr)
        elif(hit_obj.material=="reflection"):
            kr = fresnel(d, n, hit_obj.ior);
            reflection_direction = reflect(d, n);
            reflection_ray_orig = hit_point + n * epsilon if(np.dot(reflection_direction, n) < 0) else\
                                            hit_point - n * epsilon
            hit_color = cast_ray(reflection_ray_orig, reflection_direction, obj, lights, depth + 1) * kr
        else:
            #diffuse and glossy材质
            light_amt = 0
            specular_color = 0
            intensity=np.array([0.5]*3)
            shadow_point_orig = hit_point + n * epsilon if(np.dot(d, n) < 0) else\
                                        hit_point - n * epsilon
            # print(">>", d, n, np.dot(d, n))
            for light in lights:
                light_dir=light-hit_point
                light_distance=np.dot(light_dir,light_dir)
                light_dir=normalize(light_dir)
                LdotN = max(0., np.dot(light_dir, n))
                #print("<", light_dir, n)
                shadow_res = trace(shadow_point_orig, light_dir, obj)
                # print(shadow_res, shadow_point_orig, light_dir)
                if(shadow_res[0]==None):
                    inshadow=False
                else:
                    f = True
                    # print(shadow_res[0] * shadow_res[0], light_distance)
                    inshadow=(shadow_res[0] * shadow_res[0] < light_distance)
                # if inshadow:
                #     print(">>>>>>>>>>>>>>>>>>>>>>")
                light_amt += 0 if inshadow else intensity * LdotN
                # print(intensity, LdotN)
                reflection_direction = reflect(-light_dir, n)
                # print(reflection_direction)
                specular_color += pow(max(0., -np.dot(reflection_direction, d)),\
                    hit_obj.specular_exponent) * intensity
                # print(specular_color)
            # print(hit_obj.eval_diffusecolor(st))
            hit_color = light_amt * hit_obj.eval_diffusecolor(st) * hit_obj.kd + specular_color * hit_obj.ks
            # print(hit_color)
        # f = True
    return hit_color#, f 

def update_progress(progress):
    #输出进度条
    barwidth = 70
    print("[",end='')
    pos = int(barwidth * progress)
    for i in range(barwidth):
        if (i < pos):
            print("=",end='')
        elif (i == pos):
            print(">",end='')
        else:
            print(" ",end='')
    print("] %d %%\r"%(int(progress*100.)),end='')


if __name__ == "__main__":
    sphere1 = Sphere(np.array([-1, 0, -12]),2)
    sphere1.diffuse_color = np.array([0.6, 0.7, 0.8])

    sphere2 = Sphere(np.array([0.5, -0.5, -8]),1.5)
    sphere2.ior=1.5
    sphere2.material="reflection and refraction"

    v=np.array([[-5,-3,-6], [5,-3,-6], [5,-3,-16], [-5,-3,-16]])
    vindex=np.array([0, 1, 3, 1, 2, 3])
    st=np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    mesh= MeshTriangle(v,vindex,st)

    #添加三个物体,两个是球体,一个是平面
    obj=[sphere1,sphere2,mesh]
    # obj=[sphere1,mesh]
    # obj=[mesh]

    #添加两个光源
    lights=[np.array([-20, 70, 20]),np.array([30, 50, -12])]

    #frame_buffer保存计算出来的每个像素的颜色值
    # width, height = 1280, 960
    width, height = 320, 240
    frame_buffer=np.zeros((height,width,3),np.float32)
    #视角为90°
    fov=90
    #缩放比例
    scale=np.tan(deg2rad(fov*0.5))
    #宽高比
    image_aspect_ratio=width/height
    #视点
    eye_pos=np.array([0.,0.,0.])
    # xy = np.zeros((height, width, 2))
    for j in range(height):
        for i in range(width):
            if i != 158 or j != 108:
                continue
            #将[-width/2,width/2],[-height/2,height/2]映射到[-0.5,0.5],[-0.5,0.5]区域后再缩放
            x=((0.5+i)/(width/2.)-1)*scale*image_aspect_ratio
            y=((-0.5-j)/(height/2.)+1)*scale
            # xy[j, i, :] = np.array([x, y])
            #光线方向为(x,y,-1)
            d = normalize(np.array([x, y, -1]))
            #发出起点为视点,方向为d的光线,已走的路程为0,进行光线追踪的计算
            frame_buffer[j][i] = cast_ray(eye_pos, d, obj, lights, 0)
            # if f:
            print("[{}, {}] >> {}".format(j, i, frame_buffer[j][i]))
        #安慰剂按钮,显示进度条
        update_progress(j/height)
    # print(xy)
    dst=np.zeros((height,width,3),np.uint8)
    frame_buffer[frame_buffer<0.]=0.
    frame_buffer=frame_buffer*255
    cv2.convertScaleAbs(frame_buffer,dst,1.0)
    dst=cv2.cvtColor(dst,cv2.COLOR_RGB2BGR)
    cv2.imwrite("./r.png",dst)
