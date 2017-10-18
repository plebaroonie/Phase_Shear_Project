# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 13:46:31 2017

@author: Matthew Gray

version 1.1 averages the image instead of just taking a slice. This should 
allow for a y offset on the data
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.optimize import curve_fit

spacing = 0.05
ranges = 5

def function_to_fit(x, A, x0_G, sigma, k, x0_S):
    gauss = A*np.exp(-((x - x0_G)**2)/(2*sigma**2))
    sine = 0.5*(1 + np.sin(k*(x - x0_S)))
    curve = gauss*sine
    return curve
    
A = 1
sin_phi = 0 * np.pi
theta = -0 * np.pi
sigma_y = 3
sigma_x = 3
x0_G = 0
y0_G = 0
x0_S = 0
y0_S = 0
a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2);
b = np.sin(2*theta)/(4*sigma_x**2) - np.sin(2*theta)/(4*sigma_y**2);
c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2);

X = np.arange(-ranges, ranges + 1, spacing)
Y = np.arange(-ranges, ranges + 1, spacing)
X, Y = np.meshgrid(X, Y)

x = X[0]

   
gaussian = A*np.exp(-(a*(X-x0_G)**2 + 2*b*(X-x0_G)*(Y-y0_G) + c*(Y-y0_G)**2))
wave = 0.5*(1 + np.sin((X-x0_S)*np.cos(sin_phi) + (Y-y0_S)*np.sin(sin_phi)))

Z = gaussian*wave
z = []
size = len(X[0])
inv_size = 1/float(size)
for i in range(0, size):
    binned = 0
    for j in range(0, size):
        binned += Z[j][i]
    binned = binned*inv_size
    z.append(binned)
        
        
    



# Customize the z axis.
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_title('surface')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax = fig.add_subplot(222)
ax.plot(x,z)
ax.set_title('slice')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax = fig.add_subplot(223)
popt, pcov = curve_fit(function_to_fit, x, z)
ax.plot(x, function_to_fit(x, *popt))
ax.set_title('fit')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax = fig.add_subplot(224)
ax.plot(x,0.5*(1+(np.sin(popt[3]*(x - popt[4])))))
ax.set_title('fringe')
ax.set_xlabel('x')
ax.set_ylabel('z')
# Fine-tune figure; make subplots farther from each other.
fig.subplots_adjust(hspace = 0.6, wspace = 0.6)



plt.savefig('2d_average.jpg')
plt.show()


