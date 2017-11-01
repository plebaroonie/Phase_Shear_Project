# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:05:15 2017

@author: CXB431
"""
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl


def function_to_fit_linear_packing(data, k, theta, A, sigma, x0, y0, C):
        
    gaussian = A*np.exp(-(data[:,0]**2 + data[:,1]**2)/(2*sigma**2))
    wave = 0.5*(1 + C*np.sin(k*((data[:,0]-x0)*np.cos(theta) + (data[:,1] - y0)*np.sin(theta))))
    curve = gaussian*wave
    return curve 
    
def function_to_fit_grid_packing(data, k, theta, A, sigma, x0, y0, C):
    
    gaussian = A*np.exp(-(data[0]**2 + data[1]**2)/(2*sigma**2))
    wave = 0.5*(1 + C*np.sin(k*((data[0]-x0)*np.cos(theta) + (data[1] - y0)*np.sin(theta))))
    curve = gaussian*wave
    return curve
    
def band_pass(input_data, frequency):
    
    length = int(len(input_data)/2)
    output_data = input_data
    
    dt = abs(input_data[0][0][1] - input_data[0][0][0]) 
    df = 1.0/dt
    
    fourier_data = np.fft.fft2(input_data[2]) 
    
    for i in range(0, length):
      
        i_freq = (i*df)**2
        
        for j in range(0, length):
    
            if np.sqrt((j*df)**2 + i_freq) > frequency:
                output_data[i][j] = 0
                output_data[length + i][length + j] = 0
                
    output_data[2] = np.fft.ifft2(fourier_data)        

    return output_data

def FT_plot(data):
    
    fourier_data = np.fft.fft2(data[2])
    df = 1.0 / abs((data[0][0][1] - data[0][0][0]))
    length = int(len(data[0]) / 2)
    packed_data = fourier_data
    
    for i in range(0,length):
        for j in range(0, length):
            packed_data[i][j] = fourier_data[length + i][length + j]
    for i in range(0,length):
        for j in range(0, length):
            packed_data[i][j] = fourier_data[i][j]
    
    freq_X = np.linspace(-df*length, df*length, 2*length, True)
    freq_Y = freq_X
    freq_X, freq_Y = np.meshgrid(freq_X, freq_Y)
    
    print(len(freq_X), len(packed_data))
    label = 6
    mpl.rc('xtick', labelsize=label) 
    mpl.rc('ytick', labelsize=label)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(freq_X, freq_Y, packed_data.imag, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    
    
    ax.set_title('data')
    plt.show()
        
    
    
    

def create_cloud(N, k, theta, A, sigma, x0, y0, C, noise_scale):
    
    ranges = 20
    
    endpoint = True
    X = np.linspace(-ranges, ranges, N, endpoint)
    Y = np.linspace(-ranges, ranges, N, endpoint)
    X, Y = np.meshgrid(X, Y)
    
    data = np.array([X, Y])
    
    Z = function_to_fit_grid_packing(data, k, theta, A, sigma, x0, y0, C)
    
    data = np.array([data[0], data[1], Z])
    
    if not noise_scale == 0:
        for i in range(0, len(data[2])):
            for j in range(0, len(data[2][i])):
                data[2][i][j] = data[2][i][j] + noise_scale*(random.random() - 0.5)

    return data
    
def grid_to_linear(grid_data):

    linear_data = np.array([grid_data[0][0][0], grid_data[1][0][0], grid_data[2][0][0]])
        
    for i in range(1,len(grid_data[0])):
        for j in range(1,len(grid_data[0][i])):
            stacked = np.array([grid_data[0][i][j], grid_data[1][i][j], grid_data[2][i][j]])
            linear_data = np.vstack((linear_data, stacked))  
                
    return linear_data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    