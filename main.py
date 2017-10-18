# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 13:46:31 2017

@author: Matthew Gray
version 1.2 attempts to fit the entire 3d space instead of taking a 2d plot as was done earlier
"""
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.optimize import curve_fit
import fitting as fun


N = 200

A = 1
theta_s = 0
theta_g = 0 * np.pi
sigma_y = 5
sigma_x = 5
x0 = 0
y0 = 0
C = 1
freq = 50

grid_data = fun.create_cloud(N, theta_g, theta_s, A, sigma_x, sigma_y, x0, y0, C, 0)


grid_data = fun.bandpass(grid_data, freq)

linear_data = fun.grid_to_linear(grid_data)


label = 6
mpl.rc('xtick', labelsize=label) 
mpl.rc('ytick', labelsize=label)
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(grid_data[0], grid_data[1], grid_data[2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)


ax.set_title('add noise')

#guess = (theta, sin_phi, A, sigma_x, sigma_y, x0, y0)
ax = fig.add_subplot(122, projection='3d')
popt, pcov = curve_fit(fun.function_to_fit_linear_packing, linear_data[:,:2], linear_data[:,2])
ax.plot_surface(grid_data[0], grid_data[1], fun.function_to_fit_grid_packing(grid_data[0:2], *popt), cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

ax.set_title('fit')


fig.subplots_adjust(hspace=0.6)

plt.savefig('3d-fit.pdf')
plt.show()