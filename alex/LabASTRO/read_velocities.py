# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:54:33 2022

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

def read_data(file):
    f = open(file, "r")
    lines = f.readlines()
    velocities = []
    for line in lines:
        line = line.split()
        for velocity in line:
            velocities.append(velocity)
    f.close()
    
    return velocities

data_calib1_pm1 = [float(v) for v in read_data('data2_calib1_pm1_960411_961010.dat')]
data_calib1_pm2 = [float(v) for v in read_data('data2_calib1_pm2_960411_961010.dat')]

data_calib2_pm1 = [float(v) for v in read_data('data2_calib1_pm1_960411_961010.dat')]
data_calib2_pm2 = [float(v) for v in read_data('data2_calib1_pm2_960411_961010.dat')]

#data_calib2_pm1 = read_data('data2_calib2_pm1_960411_961010.dat')
#data_calib2_pm2 = read_data('data2_calib2_pm2_960411_961010.dat')

time = np.arange(790560) # number of data points
time = [9.5 + 20 * t for t in time] # data centered at 00h00mn9.5sec UT
#freq= [1./t for t in time]

freq = fftfreq(790560, 20)[:790560//2]

fft_data_calib1_pm1 = fft(data_calib1_pm1)
fft_data_calib1_pm2 = fft(data_calib1_pm2)

theoretical_frequencies = [0.535308, 0.679501, 0.970528, 1.115391, 1.260426, 1.403781, 1.543825, 1.681478, 1.817298, 1.953574, 2.090097, 2.225231, 2.359319, 2.492708, 2.626554, 2.761449, 2.896414, 3.031626, 3.16692, 3.302485, 3.438826, 3.575348, 3.712182, 3.849159, 3.986249, 4.123775, 4.261271, 4.398846, 4.536387, 4.673751, 4.946911, 5.095484, 5.234301, 5.373123, 5.512118, 5.65101, 5.78991, 5.928871, 6.067785, 6.206761, 6.345584, 6.48434, 6.62311, 6.761761, 7.038738, 7.176977, 7.315157, 7.453125, 7.590913, 0.891473, 1.036782, 1.182289, 1.325785, 1.468123, 1.606647, 1.743139, 1.879592, 2.015963, 2.152349, 2.28712, 2.42074, 2.554625, 2.689054, 2.824228, 2.959557, 3.094791, 3.23037, 3.366419, 3.502967, 3.639851, 3.776819, 3.913997, 4.051451, 4.189064, 4.326765, 4.464404, 4.601967, 4.739407, 4.876241, 5.023275, 5.162115, 5.301017, 5.44002, 5.579044, 5.718011, 5.857005, 5.996009, 6.135015, 6.273986, 0.957193, 1.101802, 1.246902, 1.390052, 1.530353, 1.668165, 1.804061, 1.940584, 2.077293, 2.212784, 2.34722, 2.480858, 2.615009, 2.750181, 2.885445, 3.02098, 3.156551, 3.292407, 3.42903, 3.565824, 3.702942, 3.840181, 3.977531, 4.115312, 4.253053, 4.390873, 4.528647, 4.66624, 4.803671, 4.939921, 5.088536, 5.227532, 5.366524, 5.505683, 5.644735, 5.783787, 5.922895, 6.06195, 6.201058, 0.864609, 1.012056, 1.158037, 1.302591, 1.445901, 1.585199, 1.722534, 1.859199, 1.995954, 2.132971, 2.268246, 2.402443, 2.536677, 2.671394, 2.807062, 2.942791, 3.078497, 3.214521, 3.350926, 3.487938, 3.625222, 3.762608, 3.900194, 4.037984, 4.175992, 4.314043, 4.452032, 4.589953, 4.727719, 4.864958, 4.999742, 5.151218, 5.290399, 5.429645, 5.56893, 5.708133, 5.847348, 5.986578, 6.125788, 6.264974]

"""plt.scatter(data_calib1_pm1, data_calib1_pm2, s=1, alpha=0.1)

plt.show()

plt.scatter(data_calib2_pm1, data_calib2_pm2, s=1, alpha=0.1)

plt.show()

plt.scatter(data_calib1_pm1, data_calib2_pm1, s=1, alpha=0.1)

plt.show()

plt.scatter(data_calib1_pm2, data_calib2_pm2, s=1, alpha=0.1)
plt.show()"""

"""plt.scatter(time, data_calib1_pm1, s=1, alpha=0.1)

plt.scatter(time, data_calib1_pm2, s=1, alpha=0.1)
plt.show()

plt.scatter(time, data_calib2_pm1, s=1, alpha=0.2)

plt.scatter(time, data_calib2_pm2, s=1, alpha=0.2)
plt.show()"""

plt.plot(freq*1e6, 2.0/790560 * np.abs(fft_data_calib1_pm1)[0:790560//2], color="black", linewidth=0.5)
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel (r'P ($\nu$)', size=14)
plt.xlim(0, 8000)
plt.show()

plt.plot(freq*1e6, 2.0/790560 * np.abs(fft_data_calib1_pm2)[0:790560//2], color="black", linewidth=0.5)
plt.plot([1e3*f for f in theoretical_frequencies], 0.08*np.random.random(168), color="red", linewidth=0.5)
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel (r'P ($\nu$)', size=14)
plt.xlim(0, 8000)
plt.show()

"""plt.plot(freq*1e6, 2.0/790560 * np.abs(fft_data_calib1_pm1)[0:790560//2], color="black", linewidth=0.5)
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel (r'P ($\nu$)', size=14)
plt.xlim(0, 100)
plt.show()"""

freq_mhz = [1e3 * f for f in freq if ((f * 1e3 > 0.5) & (f * 1e3 < 7.6))]

freq_mhz = [round(f, 6) for f in freq_mhz]

freq_mhz = np.unique(freq_mhz)

print(len(freq_mhz))

#print(freq_mhz)
print(freq_mhz)

#print(fft.fftfreq(data_calib1_pm1))