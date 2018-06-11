#LINEAR REGRESSOR
# 

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat
import time

# simple pendelum data
data = pd.read_csv("regr01.txt", sep=" ", header=None, names=['l', 't'])
print(data.head())
print(data.tail())


l = data['l'].values
t = data['t'].values
tsq = t * t


# The essenial parts of the Gradient Descent method:
# 
#  $y = mx + c$
#  
#  $E$ =$\frac{1}{n}$   $\sum_{i=1}^n (y_i - y)^2$
#  
#  $\frac{\partial E }{\partial m}$ = $\frac{2}{n}$   $\sum_{i=1}^n  -x_i(y_i - (mx_i + c))$
#  
#  $\frac{\partial E}{\partial c}$ = $\frac{2}{n}$   $\sum_{i=1}^n  -(y_i - (mx_i + c))$

def train(x, y, m, c, eta):
    const = - 2.0/len(y)
    ycalc = m * x + c
    delta_m = const * sum(x * (y - ycalc))
    delta_c = const * sum(y - ycalc)
    m = m - delta_m * eta
    c = c - delta_c * eta
    error = sum((y - ycalc)**2)/len(y)
    return m, c, error

def train_on_all(x, y, m, c, eta, iterations=1000):
    for steps in range(iterations):
        m, c, err = train(x, y, m, c, eta)
    return m, c, err


# ## TRAIN
# 
# ## Let us visualize the training:
# ### $\eta$ = 0.01
# 
# Training for 1000 iterations, plotting after every 100 iterations:

# Init m, c
m, c = 0, 0

# Learning rate
lr = 0.01

# Training for 1000 iterations, plotting after every 100 iterations:
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
plt.ion()
fig.show()
fig.canvas.draw()

for num in range(10):
    m, c, error = train_on_all(l, tsq, m, c, lr, iterations=100)
    print("m = {0:.6} c = {1:.6} Error = {2:.6}".format(m, c, error))
    y = m * l + c
    ax.clear()
    ax.plot(l, tsq, '.k')
    ax.plot(l, y)
    fig.canvas.draw()
    time.sleep(1)

# ## Plotting error vs iterations
# 
# So far we have seen how the Gradient Descent works by looking at the fit of the regression line. Let us change perspectives and plot the error at various stages. This just shows that the process is converging and gives us a feel for the rate at which it is converging.
# 
# $E = \frac{1}{n} ∑_{i=1}^n(y_i−y)^2$
# 
# $ = \frac{1}{n} ∑_{i=1}^n(y_i - mx_i - c)^2$

ms, cs,errs = [], [], []
m, c = 0, 0
eta = 0.001
for times in range(200):
    m, c, error = train_on_all(l, tsq, m, c, eta, iterations=100) # We will plot the value of for every 100 iterations
    ms.append(m)
    cs.append(c)
    errs.append(error)
epochs = range(0, 20000,100)
plt.figure(figsize=(8,5))
plt.plot(epochs, errs)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Vanilla Gradient Descent")
plt.show()

# We see that the error at saturation is around 0.01.

# ## Error vs m, c

# Let us visualize the error as a function of **m** and **c**


def error(x,y,m,c):
    ycalc = m * x + c
    error = sum((y - ycalc)**2) / len(y)
    return error

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ms1 = np.arange(-20, 20, 0.1)
cs1 = np.arange(-20, 20, 0.1)
X, Y = np.meshgrid(ms1, cs1)
err = []
for i in range(len(ms1)):
    for j in range(len(cs1)):
        err.append(error(l,tsq,ms1[i],cs1[j]))
err = np.array(err)
Z = np.reshape(err,(-1,len(ms1)))
print(X.shape, Y.shape, Z.shape)
ax.plot_surface(X, Y, Z) 
ax.set_xlabel('m')
ax.set_ylabel('c')
ax.set_zlabel('error')
plt.show()

