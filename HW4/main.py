import matplotlib.pyplot as plt 
import numpy as np 
import argparse


def C_i_within_n(n, i):
    a = 1 
    m = n 
    while m > n-i:
        a = a*m 
        m -= 1
    b = 1
    if i <= 1:
        b = 1
    else:
        j = i 
        while j > 0:
            b = b*j
            j -= 1
    return a // b 


def get_C_coefficients(n):
    R = []
    for i in range(n+1):
        R.append(C_i_within_n(n, i))
    return R 


def naive_beizer(points):
    n = len(points)-1
    points = np.array(points)
    
    ts = np.linspace(0., 1., 1000)
    C_coefficients = get_C_coefficients(n) # 计算 C_n^i
    C_coefficients = np.array(C_coefficients)
    curve = []
    for t in ts:
        t_power_i = [t**i for i in range(n+1)]
        complementary_power_n_minus_i = [(1-t)**(n-i) for i in range(n+1)]
        
        t_power_i = np.array(t_power_i)
        complementary_power_n_minus_i = np.array(complementary_power_n_minus_i)
        B_coefficients = C_coefficients*t_power_i*complementary_power_n_minus_i
        B_coefficients = B_coefficients[:, np.newaxis]
        point = (points*B_coefficients).sum(axis=0)

        curve.append(point)

    return np.array(curve)


def on_key_press(event):
    key = event.key
    if key == "enter":
        args.quit = True
        plt.close()
    elif key == "d":
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW4 CMD.")
    parser.add_argument("--n_points", type=int, default=4)
    args = parser.parse_args()
    args.quit = False

    H, W = 512+256, 512+256
    title = "Bezier Curve"

    while True:

        BG = np.zeros((H, W, 3))
        plt.imshow(BG)
        ## 监听点的数目
        points = plt.ginput(args.n_points)
        plt.close()
        ## 
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("key_press_event", on_key_press)
        plt.imshow(BG)
        x0, y0 = None, None
        for point in points:
            x, y = point
            plt.plot([x], [y], 'w.')
            if x0 is not None:
                plt.plot([x0, x], [y0, y], color="white", linestyle="dashed", linewidth=0.2)
            x0 = x 
            y0 = y 

        curve = naive_beizer(points)
        plt.scatter(curve[:, 0], curve[:, 1], marker=".", c="r", s=0.2)
        plt.show()

        if args.quit == True:
            break

        ## 
        # break

    