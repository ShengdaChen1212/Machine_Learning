# training 1
initial = [1, 1]
for i in range(1, 9):
    temp = initial[i-1]+initial[i]
    initial.append(temp)
print(initial)

#%%
# training 2
import numpy as np
import matplotlib.pyplot as plt

def  plot_y(begin,end,point_number):
    x = np.arange(begin, end, point_number)
    y = pow(np.cos(x), 2) + 2 * (x ** 0.5)
    plt.figure()
    plt.plot(x, y)
    plt.show

plot_y(0, 20, 1)
plot_y(0, 20, 0.2)