## `Python`-implemented Assignments of <br/>the Hot Courseð¥*Games101*ð¥

#### 1. Assignment 1    

![screenshot-assignment-1](./assets/001.png)

#### 2. Assignment 2

![screenshot-assignment-2](./assets/003.png)

#### 3. Assignment 3ââShading

:one: å®ç° `ObjLoader`ï¼ä½åæ¥åç° `python` æç´æ¥çåºå¯ä»¥ç¨ðï¼ä¸éè¦å»åé£ä¹ç¹æç `.obj` æä»¶è§£æå¨ï¼å³

```python
## TOINSTALL
#   pip install objloader

## TOUSEIT
from objloader import Obj
obj = Obj.open(obj_pth).to_array() # å¶æ ¼å¼å·²ç»æ¯éä¸ªâ³äº

for i in range(0, len(obj), 3):    # éåæ¯ä¸ªâ³
    for v in obj[i:i+3]:           # éåæ¯ä¸ªç»ç¹
        x, y, z, nx, ny, nz, u, v, _ = v
        # xyz -> ç»ç¹çä¸çåæ 
        # nx-ny-nz -> æ³åé
        # uv -> çº¹çåæ 
        
        # ç¨ VM ç©éµå¯¹ xyz ä½æ¨¡ååæ¢
        # ç¨ [(VM)^(-1)]^T å¯¹ nx-ny-nz ä½æ¨¡ååæ¢
        # TODO
```

ç¨éæºçé¢è²å¡«åæ¯ä¸ªä¸è§å½¢åºæ¬ååï¼å¾å°ï¼

![004-1](./assets/004-1.png)

2ï¸â£ **Shader**

å¨ `python` ä¸­ï¼åè®°å¯¹é¤æ°è¦å  `epsilon (e.g., 1e-8)`ï¼ä¸ç¶å®¹æåºäºã

| ä½¿ç¨æ³åéçåéä»£æ¿ `R/G/B` çè² | ä½¿ç¨çº¹çå¾çè²<br/>å¤å ï¼ç¯å¢å+æ¼«åå°+é«å |
| --------------------------------- | ------------------------------------------- |
| ![004-3](./assets/004-3.png)      | ![004-2](./assets/004-2.png)                |
| **Phong**                         | **Bump**                                    |
| ![004-4](./assets/004-4.png)      | ![004-5](./assets/004-5.png)                |
| **Displacement**                  | **å¥½ä¸ç¹çè§å¾åä¸ªæ°**                      |
| ![004-6](./assets/004-6.png)      | ![004-2-2](./assets/004-2-2.png)            |

#### 4. Assignment 4ââç»å¶è´å¡å°æ²çº¿

**è´å¡å°æ²çº¿**

![005](./assets/005.png)

**Mesh Subdivision**

æä»¬å¯¹ç®±å­æ¨¡ååäºç½æ ¼ç»åï¼å¹¶ä½¿ç¨çº¹çå¾çè²ãè¯¦è§ `./HW4/main2.py`ã

| ![005-1](./assets/005-1.png)<br/>Iter = 0     | ![005-1](./assets/005-2.png)<br/>Iter = 1     |
| --------------------------------------------- | --------------------------------------------- |
| ![005-1](./assets/005-3.png)<br/>**Iter = 2** | ![005-1](./assets/005-4.png)<br/>**Iter = 3** |

#### 5. Assignment 5ââ'Whitted-Style' Ray Tracing

![006](./assets/006.png)

ä¸è¾¹æ¯èæ¬ `Render.py` ä¸­ç¸å³ç®æ³çç¤ºæå¾ï¼è¯¦è§ `./assets/000.pptx`

![1653013679598](./assets/007.png)

![1653013749274](./assets/008.png)

è®°å½å¶èæ¶ä¸ºï¼`364.4381 sec`.

#### 6. Assignment 6ââAcceleration Structure: BVH

The output image is being rendered, pls wait ...

![009](./assets/009.png)

## Assignment 8ââè´¨ç¹-å¼¹ç°§ç³»ç»

<img src="./assets/010.gif" style="width:384px; align:center;"/>

