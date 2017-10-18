# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:05:15 2017

@author: CXB431
"""
import numpy as np

def rotation_coefficients(theta, sigma_x, sigma_y):
    a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
    b = np.sin(2*theta)/(4*sigma_x**2) - np.sin(2*theta)/(4*sigma_y**2)
    c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
    
    return a,b,c

def function_to_fit_linear_packing(data, theta_g, theta_s, A, sigma_x, sigma_y, x0, y0):
    
    a, b, c = rotation_coefficients(theta_g, sigma_x, sigma_y)
    
    gaussian = A*np.exp(-(a*(data[:,0]-x0)**2 + 2*b*(data[:,0]-x0)*(data[:,1]-y0) + c*(data[:,1]-y0)**2))

    wave = 1 + np.sin(data[:,0]*np.cos(theta_s) + data[:,1]*np.sin(theta_s))
    curve = 0.5*gaussian*wave
    return curve 
    
def function_to_fit_grid_packing(data, theta_g, theta_s, A, sigma_x, sigma_y, x0, y0):
    
    a, b, c = rotation_coefficients(theta_g, sigma_x, sigma_y)
    
    gaussian = A*np.exp(-(a*(data[0]-x0)**2 + 2*b*(data[0]-x0)*(data[1]-y0) + c*(data[1]-y0)**2))

    wave = 1 + np.sin(data[0]*np.cos(theta_s) + data[1]*np.sin(theta_s))
    curve = 0.5*gaussian*wave
    return curve
    
def band_pass(input_data, frequency, df):
    
    length = int(len(input_data)/2)
    output_data = input_data
    
    for i in range(0, length):
      
        i_freq = (i*df)**2
        
        for j in range(0, length):
    
            if np.sqrt((j*df)**2 + i_freq) > frequency:
                output_data[i][j] = 0
                output_data[length + i][length + j] = 0
            
    
    output_data
    return output_data