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
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.optimize import curve_fit
import fitting as fun


N = 200

A = 1
theta = np.pi/4
sigma = 9
x0 = -0.5*np.pi
y0 = 0
C = 1
freq = 50
k = 0.4

grid_data = fun.create_cloud(N, k, theta, A, sigma, x0, y0, C, 0)

#grid_data = fun.band_pass(grid_data, freq)

linear_data = fun.grid_to_linear(grid_data)





label = 6
mpl.rc('xtick', labelsize=label) 
mpl.rc('ytick', labelsize=label)
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(grid_data[0], grid_data[1], grid_data[2], cmap=cm.inferno,
                       linewidth=0, antialiased=True)

ax._axis3don = False
ax.set_title('data')

#guess = (theta, sin_phi, A, sigma_x, sigma_y, x0, y0)
lower = [0, -0.5*np.pi, 0, 0, -np.pi, -np.pi, 0]
upper = [np.inf, 0.5*np.pi, 1, np.inf, np.pi, np.pi, 1]
ax = fig.add_subplot(122, projection='3d')
popt, pcov = curve_fit(fun.function_to_fit_linear_packing, linear_data[:,:2], linear_data[:,2], bounds=(lower, upper))
print(k, theta, A, sigma, x0, y0, C)
print(popt)
ax.plot_surface(grid_data[0], grid_data[1], fun.function_to_fit_grid_packing(grid_data[0:2], *popt), cmap=cm.inferno,
                       linewidth=0, antialiased=True)

ax.set_title('fit')

ax._axis3don = False


fig.subplots_adjust(hspace=-0.6)

plt.savefig('3d-fit.pdf')
plt.show()
