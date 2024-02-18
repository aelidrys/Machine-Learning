import numpy as np
import matplotlib.pyplot as plt

m = (25-22)/(38-22)
c = 25-(38*m)
X = [1  ,5  ,15 ,22 ,27 ,38 ,41 ,50]
Y = [12 ,15 ,13 ,22 ,30 ,25 ,31 ,40]

# m: slope and c: intresection with y
def error_count(m, c):
    sum_er = 0
    for (x, y_gt) in zip(X,Y):
        y_pd = m * x + c
        sum_er += pow(y_pd - y_gt, 2)
    return (sum_er / len(Y) / 2)

# error = error_count(m, c)  
# print("m = ", "{:.2f}".format(m), "\nc = ", "{:.2f}".format(c), "\nerror = ", "{:.2f}".format(error))

# line_y = np.array(X) * m + c

# plt.plot(X,Y, marker='o', label='current points')
# plt.plot(X,line_y, label='prediction line',color='red')
# plt.title('MSE')
# plt.xlabel(X)
# plt.ylabel(Y)
# plt.legend()

# plt.show()

def fun(x):
    return 3 * x ** 2 + 4 * x + 7

def f_deriv(x):
    return 6 * x + 4

def g_descent(d_f,init_x, l_r = 0.01, precision = 0.0001):
    cur_x = init_x
    last_x = float('inf')
    x_list = [cur_x]

    while abs(cur_x - last_x) > precision:
        last_x = cur_x
        cur_x -= d_f(cur_x) * l_r
        x_list.append(cur_x)
    print(f'the minimum x: {cur_x}')
    return x_list


def visuale(x_list,fun,range_start, range_end):
    x_set = np.linspace(range_start, range_end, 100)
    y_set = fun(x_set)

    plt.plot(x_set, y_set, label="f(x)")
    plt.title("gresient descent of 3x^2 + 4x + 7")
    plt.xlabel('X')
    plt.ylabel('Y')

    for i, xp in enumerate(x_list[::3]):
        yp = fun(xp)
        color = 'ro' if i%2 else 'bo'
        plt.plot(xp,yp,color)

    plt.show()

print("f(1) = ",fun(2))
print("f'(1) = ",f_deriv(2))

x_list = g_descent(f_deriv, -6)

visuale(x_list,fun,-8,7)

