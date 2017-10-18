# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:05:15 2017

@author: CXB431
"""
import numpy as np
import random

def rotation_coefficients(theta, sigma_x, sigma_y):
    a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
    b = np.sin(2*theta)/(4*sigma_x**2) - np.sin(2*theta)/(4*sigma_y**2)
    c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
    
    return a,b,c

def function_to_fit_linear_packing(data, theta_g, theta_s, A, sigma_x, sigma_y, x0, y0, C):
    
    a, b, c = rotation_coefficients(theta_g, sigma_x, sigma_y)
    
    gaussian = A*np.exp(-(a*(data[:,0])**2 + 2*b*(data[:,0])*(data[:,1]) + c*(data[:,1])**2))
    wave = 0.5*(1 + C*np.sin((data[:,0]-x0)*np.cos(theta_s) + (data[:,1] - y0)*np.sin(theta_s)))
    curve = gaussian*wave
    return curve 
    
def function_to_fit_grid_packing(data, theta_g, theta_s, A, sigma_x, sigma_y, x0, y0, C):
    
    a, b, c = rotation_coefficients(theta_g, sigma_x, sigma_y)
    
    gaussian = A*np.exp(-(a*(data[0])**2 + 2*b*(data[0])*(data[1]) + c*(data[1])**2))
    wave = 0.5*(1 + C*np.sin((data[0]-x0)*np.cos(theta_s) + (data[1] - y0)*np.sin(theta_s)))
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

def create_cloud(N, theta_g, theta_s, A, sigma_x, sigma_y, x0, y0, c, noise_scale):
    
    ranges = 20
    
    endpoint = True
    X = np.linspace(-ranges, ranges, N, endpoint)
    Y = np.linspace(-ranges, ranges, N, endpoint)
    X, Y = np.meshgrid(X, Y)
    
    data = np.array([X, Y])
    Z = function_to_fit_grid_packing(data, theta_g, theta_s, A, sigma_x, sigma_y, x0, y0, c)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    