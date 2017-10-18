# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 13:46:31 2017

@author: Matthew Gray
version 1.2 attempts to fit the entire 3d space instead of taking a 2d plot as was done earlier
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.optimize import curve_fit
import random

import functions as fun

label = 6
spacing = 0.1
ranges = 10

A = 1
theta_s = 0
theta_g = -0 * np.pi
sigma_y = 5
sigma_x = 5
x0 = 0
y0 = 0

a, b, c = fun.rotation_coefficients(theta_g, sigma_x, sigma_y)

X = np.arange(-ranges, ranges + 1, spacing)
Y = np.arange(-ranges, ranges + 1, spacing)
X, Y = np.meshgrid(X, Y)

x = X[0]
find = -1;
found = False;
while not found:
    find+=1
    found = (abs(Y[find][0]) < 0.05)

    



gaussian = A*np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
wave = 1 + np.sin(X*np.cos(theta_s) + Y*np.sin(theta_s))

Z = 0.5*gaussian*wave
mpl.rc('xtick', labelsize=label) 
mpl.rc('ytick', labelsize=label)
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
                       
ax.set_xlim([-ranges, ranges])
ax.set_ylim([-ranges, ranges])
ax.set_title('created fringe')

z = Z[find]
 
data1 = np.array([X[0][0], Y[0][0], Z[0][0]])

data2 = np.array([X, Y, Z])

for i in range(0, len(data2[2])):
    for j in range(0, len(data2[2][i])):
        random = random
        data2[2][i][j] = data2[2][i][j] + 10*(random.random() - 0.5)

# Customize the z axis.

data1 = np.array([X[0][0], Y[0][0], Z[0][0]])
 
fourier_data = np.fft.fft2(data2[2])
fourier_data = fun.band_pass(fourier_data, 50, 1/spacing)
data2[2] = np.fft.ifft2(fourier_data)

for i in range(1,len(X)):
    for j in range(1,len(X[0])):
       data1 = np.vstack((data1, np.array([data2[0][i][j], data2[1][i][j], data2[2][i][j]])))

ax = fig.add_subplot(222, projection='3d')
ax.plot_surface(data2[0], data2[1], data2[2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
                       
ax.set_xlim([-ranges, ranges])
ax.set_ylim([-ranges, ranges])

ax.set_title('add noise')
mpl.rc('xtick', labelsize=label) 
mpl.rc('ytick', labelsize=label)

#guess = (theta, sin_phi, A, sigma_x, sigma_y, x0, y0)
ax = fig.add_subplot(223, projection='3d')
popt, pcov = curve_fit(fun.function_to_fit_linear_packing, data1[:,:2], data1[:,2])
ax.plot_surface(data2[0], data2[1], fun.function_to_fit_grid_packing(data2[0:2], *popt), cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
ax.set_xlim([-ranges, ranges])
ax.set_ylim([-ranges, ranges])
ax.set_title('fit')
mpl.rc('xtick', labelsize=label) 
mpl.rc('ytick', labelsize=label)

ax = fig.add_subplot(224, projection='3d')
ax.plot_surface(data2[0], data2[1], Z - fun.function_to_fit_grid_packing(data2[0:2], *popt), cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
ax.set_xlim([-ranges, ranges])
ax.set_ylim([-ranges, ranges])
ax.set_title('difference')
mpl.rc('xtick', labelsize=label) 
mpl.rc('ytick', labelsize=label)
fig.subplots_adjust(hspace=0.6)

plt.savefig('3d-fit.pdf')
plt.show()