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
phi0 = np.pi * 0.5
C = 1
freq = 50
k = 0.4
noise = 0

grid_data = fun.create_cloud(N, k, theta, A, sigma, phi0, C, noise)

#grid_data = fun.band_pass(grid_data, freq)

linear_data = fun.grid_to_linear(grid_data)

name = np.array(['k', 'theta', 'A', 'sigma', 'phi0', 'C'])
param = np.array([k, theta, A, sigma, phi0, C])


label = 8
mpl.rc('xtick', labelsize=label) 
mpl.rc('ytick', labelsize=label)
fig = plt.figure()


ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(grid_data[0], grid_data[1], grid_data[2], cmap=cm.inferno,
                       linewidth=0, antialiased=True)
#ax._axis3don = False

ax.view_init(30, 110)





#guess = (theta, sin_phi, A, sigma_x, sigma_y, x0, y0)
lower = [0, -0.5*np.pi, 0, 0, 0, 0]
upper = [np.inf, 0.5*np.pi, 1, np.inf, 2*np.pi, 1]
ax = fig.add_subplot(122, projection='3d')
popt, pcov = curve_fit(fun.function_to_fit_linear_packing, linear_data[:,:2], linear_data[:,2], bounds=(lower, upper))
perr = np.sqrt(np.diag(pcov))

file = open("params_noise_{0:.1g}.txt".format(noise), "w")

file.write("param & value & found & std & error\\\\\n")
for i in range(0, len(param)):
    print("{0:s} & {1:.3g} & {2:.3g} & {3:.1g} & {4:.1g}\\\\\n".format(name[i],param[i], popt[i], perr[i], abs(param[i] - popt[i])))



ax.plot_surface(grid_data[0], grid_data[1], fun.function_to_fit_grid_packing(grid_data[0:2], *popt), cmap=cm.inferno,
                       linewidth=0, antialiased=True)

#ax.set_title('fit')

ax.set_xlabel('$y$', size = '16')
ax.set_ylabel('$z$', size = '16')
ax.set_zlabel('$P$', size = '16')

ax._axis3don = True
ax.view_init(30, 110)

fig.subplots_adjust(hspace=-0.6)

plt.savefig('data.pdf')
plt.show()
file.close