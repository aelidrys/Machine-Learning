import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def f_xy(x, y):
    return 2 * x ** 2 - (4 * x * y) + y ** 4 + 2

def f_dx(x, y):
    return 4 * x - 4 * y

def f_dy(x, y):
    return 4 * y ** 3 - 4 * x

def g_descent(f_dx, f_dy, i_x, i_y, l_r = 0.01, precision = 0.0001):
    cur_xy = np.array([i_x, i_y])
    last_xy = np.array([float('inf'), float('inf')])
    xy_list = [cur_xy]
    it = 0

    while norm(cur_xy - last_xy) > precision and it < 100:
        last_xy = cur_xy.copy()
        gr_x = f_dx(cur_xy[0], cur_xy[1]) * l_r
        gr_y = f_dy(cur_xy[0], cur_xy[1]) * l_r
        gr_xy = np.array([gr_x, gr_y])
        cur_xy -= gr_xy
        xy_list.append(cur_xy)
        it += 1
    print(f'the minimum (x, y): ({cur_xy[0]}, {cur_xy[1]})')
    return xy_list

print("f_xy(3, 2) =", f_xy(3, 2))
print("f_dx(3, 2) =", f_dx(3, 2))
print("f_dy(3, 2) =", f_dy(3, 2))
 = g_descent(f_dx, f_dy, -5, 0.1)
