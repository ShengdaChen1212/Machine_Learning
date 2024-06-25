# problem 1
List = []
for i in range(100, 501):
    if ((i//100) ** 3 + ((i//10) % 10) ** 3 + (i % 10) ** 3) == i:
        List.append(i)
print(List)

# %%
# problem 2
def prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


prime_num = []
i = 2
while i < 50:
    prime_check = prime(i)
    if prime_check:
        prime_num.append(i)
    i = i + 1
print(prime_num)

# %%
# problem 3
import numpy as np

card = np.zeros((9*8*7, 3))
count = 0
total = 0

for i in range(1, 10):
    for j in range(1, 10):
        for k in range(1, 10):
            if (i != j) & (i != k) & (j != k):
                card[total, :] = (i, j, k)
                total = total + 1
            if (i > j > k):
                count += 1

prob = count/total * 100
print(card)
print("The probability is :", prob, "%")

# %%
# problem 4

def reverse(nums):
    answer = []
    positive = []
    negative = []
    for i in range(len(nums)):
        if nums[i] > 0:
            positive.append(nums[i])
        elif nums[i] < 0:
            negative.append(nums[i])

    for i in range(len(nums)//2):
        answer.append(positive[i])
        answer.append(negative[i])
    return answer


a = [3, 1, -2, -5, 2, -4]
b = [-1, 1]
print(reverse(a))
print(reverse(b))

# %%
# problem 5
from numpy import random
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

Data = sio.loadmat('moon.mat')
coordinate = Data['moon']

x = coordinate[:, 0]
y = coordinate[:, 1]

def simple_linear_regression(raw_x, raw_y):
    n = np.size(raw_x)
    x = np.array(raw_x)
    y = np.array(raw_y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num1 = np.sum(y*x) - n * y_mean * x_mean
    num2 = np.sum(x*x) - n * x_mean * x_mean

    b_1 = num1 / num2
    b_0 = y_mean - b_1 * x_mean

    return (b_0, b_1)

b0, b1 = simple_linear_regression(x, y)
bfl_x = x
bfl_y = b0 + b1 * bfl_x
Y = 'y = ' + str(b0) + str(b1) + ' * x'

plt.figure()
plt.scatter(x, y)
plt.title("110011141")
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.plot(bfl_x, bfl_y, color="r", label=Y)
plt.legend(loc='upper right', shadow=True)
plt.show()

# %%
# problem 6
from numpy import random
import numpy as np
import matplotlib.pyplot as plt

class coins:
    def __init__(self, N, P):    # constructor
        self.h = random.binomial(n=N, p=P, size=100000)
        self.count = {}
        self.times = 0
        standard = np.std(self.h)
        average = np.mean(self.h)
        upper = average + standard
        lower = average - standard
        # counting each number's frequency
        for i in self.h:
            if i in self.count:
                self.count[i] += 1
            else:
                self.count[i] = 1

        for j in self.h:
            if (j < upper) & (j > lower):
                self.times += 1

    def plot(self):
        plt.figure()
        plt.bar(x=list(self.count.keys()), height=list(self.count.values()))
        plt.show()

    def percent(self):
        print("The percentage of the data within a standard deviation is : ",
              self.times/100000*100, "%")


coin_1 = coins(40, 0.5)
coin_1.plot()
coin_1.percent()

coin_2 = coins(40, 0.8)
coin_2.plot()
coin_2.percent()

coin_3 = coins(20, 0.5)
coin_3.plot()
coin_3.percent()

# %%
# problem 7
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-4, 4, 0.001)
X = np.zeros(2000)
X = np.append(X, np.arange(-1, 0, 0.001))
X = np.append(X, np.ones(1000))
X = np.append(X, (np.ones(1000)*2))
X = np.append(X, np.arange(1, 0, -0.001))
X = np.append(X, np.zeros(2000))

plt.figure()
plt.step(t, X)
plt.ylim([-4,4])
plt.grid()
plt.xlabel('t')
plt.ylabel('X(t)')
plt.show()

Y = np.flip(X)
T = t + 2
plt.figure()
plt.step(T, Y)
plt.ylim([-4,4])
plt.grid()
plt.xlabel('t')
plt.ylabel('X(2 - t)')
plt.show()

# %%
# problem 8
from calculator import Calculator

calculation = Calculator(5, 2)
a = calculation.power()
b = calculation.addPower(10)
print("a :", a)
print("b :", b)
