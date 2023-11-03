# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:54:33 2022

@author: alexa
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from lmfit.models import LorentzianModel, LinearModel, QuadraticModel # requires lmfit

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

t0 = 2450184.50011 # JD

time_JD = [t + t0 for t in time]

"""plt.scatter(time_JD, data_calib1_pm1, s=1, alpha=0.1, color="black")
plt.xlabel(r'Time (JD)', size=14)
plt.ylabel (r'Velocity (m/s)', size=14)
plt.show()

plt.scatter(time_JD, data_calib1_pm2, s=1, alpha=0.1, color="black")
plt.xlabel(r'Time (JD)', size=14)
plt.ylabel (r'Radial Velocity (m/s)', size=14)
plt.show()

plt.scatter(data_calib1_pm1, data_calib1_pm2, s=1, alpha=0.1, color="black")
model = LinearModel()
params = model.make_params()
result = model.fit(data_calib1_pm2, params, x=data_calib1_pm1)
print(result.fit_report())
plt.plot(data_calib1_pm1, result.best_fit, label='Best Fit', color ="darkred", ls="dashed")
plt.plot(data_calib1_pm1, data_calib1_pm1, label='Maximum Correlation', color ="royalblue", ls="dashed")
plt.xlabel(r'Radial Velocity PM1 (m/s)', size=14)
plt.ylabel (r'Radial Velocity PM2 (m/s)', size=14)
plt.legend(loc='upper left')"""
plt.show()

time_min = [t/60 for t in time]

plt.plot(time_min[:180], data_calib1_pm1[:180], color="black")
plt.plot(time_min[:180], data_calib1_pm2[:180], color="darkslategray", alpha=0.8)
plt.xlabel(r'Time (min)', size=14)
plt.ylabel (r'Radial Velocity (m/s)', size=14)
plt.show()

"""plt.scatter(time, data_calib1_pm2, s=1, alpha=0.1)
plt.show()

plt.scatter(time, data_calib2_pm1, s=1, alpha=0.2)
plt.show()

plt.scatter(time, data_calib2_pm2, s=1, alpha=0.2)
plt.show()"""

tfl0 = [0.535308, 0.679501, 0.970528, 1.115391, 1.260426, 1.403781, 1.543825, 1.681478, 1.817298, 1.953574, 2.090097, 2.225231, 2.359319, 2.492708, 2.626554, 2.761449, 2.896414, 3.031626, 3.16692 , 3.302485, 3.438826, 3.575348, 3.712182, 3.849159, 3.986249, 4.123775, 4.261271, 4.398846, 4.536387, 4.673751, 4.946911, 5.095484, 5.234301, 5.373123, 5.512118, 5.65101 , 5.78991 , 5.928871, 6.067785, 6.206761, 6.345584, 6.48434, 6.62311 , 6.761761, 7.038738, 7.176977, 7.315157, 7.453125, 7.590913]
tfl1 = [0.891473, 1.036782, 1.182289, 1.325785, 1.468123, 1.606647, 1.743139,
 1.879592, 2.015963, 2.152349, 2.28712 , 2.42074 , 2.554625, 2.689054,
 2.824228, 2.959557, 3.094791, 3.23037 , 3.366419, 3.502967, 3.639851,
 3.776819, 3.913997, 4.051451, 4.189064, 4.326765, 4.464404, 4.601967,
 4.739407, 4.876241, 5.023275, 5.162115, 5.301017, 5.44002 , 5.579044,
 5.718011, 5.857005, 5.996009, 6.135015, 6.273986]
tfl2 = [0.957193, 1.101802, 1.246902, 1.390052, 1.530353, 1.668165, 1.804061,
 1.940584, 2.077293, 2.212784, 2.34722 , 2.480858, 2.615009, 2.750181,
 2.885445, 3.02098 , 3.156551, 3.292407, 3.42903 , 3.565824, 3.702942,
 3.840181, 3.977531, 4.115312, 4.253053, 4.390873, 4.528647, 4.66624 ,
 4.803671, 4.939921, 5.088536, 5.227532, 5.366524, 5.505683, 5.644735,
 5.783787, 5.922895, 6.06195 , 6.201058]
tfl3 = [0.864609, 1.012056, 1.158037, 1.302591, 1.445901, 1.585199, 1.722534,
 1.859199, 1.995954, 2.132971, 2.268246, 2.402443, 2.536677, 2.671394,
 2.807062, 2.942791, 3.078497, 3.214521, 3.350926, 3.487938, 3.625222,
 3.762608, 3.900194, 4.037984, 4.175992, 4.314043, 4.452032, 4.589953,
 4.727719, 4.864958, 4.999742, 5.151218, 5.290399, 5.429645, 5.56893 ,
 5.708133, 5.847348, 5.986578, 6.125788, 6.264974]

def plot_power_spectrum(pm, xmin, xmax, scale=None):
    plt.plot(freq*1e6, 2.0/790560 * np.abs(pm)[0:790560//2], color="black", linewidth=0.5)
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'P ($\nu$) (m$^2$s$^{-2}\mu$Hz$^{-1}$)', size=14)
    #plt.ylabel (r'Power (a.u.)', size=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0.00, 0.15)
    if scale == 'log':
        plt.yscale('log')
        plt.ylim(2e-5, 1e-1)
    plt.show()
    
"""plot_power_spectrum(fft_data_calib1_pm1, 0, 1000, 'log')
plot_power_spectrum(fft_data_calib1_pm2, 0, 1000, 'log')
plot_power_spectrum(fft_data_calib1_pm1, 0, 1000)
plot_power_spectrum(fft_data_calib1_pm2, 0, 1000)
plot_power_spectrum(fft_data_calib1_pm1, 1000, 5000)
plot_power_spectrum(fft_data_calib1_pm2, 1000, 5000)"""

plt.plot(freq*1e6, 2.0/790560 * np.abs(fft_data_calib1_pm2)[0:790560//2], color="darkslategray", linewidth=0.5, alpha=0.8)
plt.plot(freq*1e6, 2.0/790560 * np.abs(fft_data_calib1_pm1)[0:790560//2], color="black", linewidth=0.5)
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel (r'P ($\nu$) (m$^2$s$^{-2}\mu$Hz$^{-1}$)', size=14)
    #plt.ylabel (r'Power (a.u.)', size=14)
plt.xlim(0, 1000)
plt.ylim(0.00, 0.15)
plt.yscale('log')
plt.ylim(2e-5, 1e-1)
plt.show()

#plot difference between spectra


colors = ['darkblue', 'darkcyan', 'darkgreen', 'olivedrab']

def compare_frequencies(l, color, figsize='large'):
    plt.plot(freq*1e6, 2.0/790560 * np.abs(fft_data_calib1_pm1)[0:790560//2], color="black", linewidth=0.5)
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'P ($\nu$)', size=14)
    plt.xlim(1000, 5000)
    plt.ylim(0.00, 0.15)
    plt.vlines([1e3 * f for f in l], 0.00, 0.15, color=color, linewidth=0.5, linestyles='dashed')
    if figsize == 'large':
        plt.figure(figsize=(16,8))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$\nu$ ($\mu$Hz)', size=16)
        plt.ylabel (r'P ($\nu$) (m$^2$s$^{-2}\mu$Hz$^{-1}$)', size=16)
        plt.legend(loc='upper right', prop={'size': 14})
    plt.show()


    plt.plot(freq*1e6, 2.0/790560 * np.abs(fft_data_calib1_pm2)[0:790560//2], color="black", linewidth=0.5)
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'P ($\nu$)', size=14)
    plt.xlim(1000, 5000)
    plt.ylim(0.00, 0.15)
    plt.vlines([1e3 * f for f in l], 0.00, 0.15, color=color, linewidth=0.5, linestyles='dashed')
    if figsize == 'large':
        plt.figure(figsize=(16,8))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$\nu$ ($\mu$Hz)', size=16)
        plt.ylabel (r'P ($\nu$) (m$^2$s$^{-2}\mu$Hz$^{-1}$)', size=16)
        plt.legend(loc='upper right', prop={'size': 14})
    plt.show()

#compare_frequencies(tfl0, "red")
#compare_frequencies(tfl1, colors[1])
#compare_frequencies(tfl2, colors[2])
#compare_frequencies(tfl3, colors[3])
#plt.scatter([1e3 * f for f in theoretical_frequencies], 2.0/len(theoretical_frequencies) * np.abs(fft([f*1e3 for f in theoretical_frequencies])), color="red")

def direct_frequency_comparision(xmin, xmax, figsize='large'):
    plt.figure(figsize=(16,8))
    plt.plot(freq*1e6, 2.0/790560 * np.abs(fft_data_calib1_pm1)[0:790560//2], color="black", linewidth=0.5)
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'P ($\nu$) (m$^2$s$^{-2}\mu$Hz$^{-1}$)', size=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0.00, 0.15)
    plt.vlines([1e3 * f for f in tfl0], 0.01, 0.15, color=colors[0], linewidth=1.5, label="l = 0")
    plt.vlines([1e3 * f for f in tfl1], 0.01, 0.15, color=colors[1], linewidth=1.5,  label="l = 1")
    plt.vlines([1e3 * f for f in tfl2], 0.01, 0.15, color=colors[2], linewidth=1.5,  label="l = 2")
    plt.vlines([1e3 * f for f in tfl3], 0.01, 0.15, color=colors[3], linewidth=1.5, label="l = 3")
    plt.legend(loc='upper right', prop={'size': 12})
    if figsize == 'large':
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$\nu$ ($\mu$Hz)', size=16)
        plt.ylabel (r'P ($\nu$) (m$^2$s$^{-2}\mu$Hz$^{-1}$)', size=16)
        plt.legend(loc='upper right', prop={'size': 14})
    plt.show()
    
    plt.figure(figsize=(16,8))
    plt.vlines([1e3 * f for f in tfl0], 0.01, 0.15, color=colors[0], linewidth=1.5,  label="l = 0")
    plt.vlines([1e3 * f for f in tfl1], 0.01, 0.15, color=colors[1], linewidth=1.5,  label="l = 1")
    plt.vlines([1e3 * f for f in tfl2], 0.01, 0.15, color=colors[2], linewidth=1.5, label="l = 2")
    plt.vlines([1e3 * f for f in tfl3], 0.01, 0.15, color=colors[3], linewidth=1.5, label="l = 3")
    plt.plot(freq*1e6, 2.0/790560 * np.abs(fft_data_calib1_pm2)[0:790560//2], color="black", linewidth=1)
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'P ($\nu$) (m$^2$s$^{-2}\mu$Hz$^{-1}$)', size=14)
    plt.xlim(xmin, xmax)
    plt.ylim(0.00, 0.15)
    plt.legend(loc='upper right', prop={'size': 12})
    if figsize == 'large':
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$\nu$ ($\mu$Hz)', size=16)
        plt.ylabel (r'P ($\nu$) (m$^2$s$^{-2}\mu$Hz$^{-1}$)', size=16)
        plt.legend(loc='upper right', prop={'size': 14})
    plt.show()

"""direct_frequency_comparision(1000, 2000)
direct_frequency_comparision(2000, 3000)
direct_frequency_comparision(3000, 4000)
direct_frequency_comparision(4000, 5000)"""

#direct_frequency_comparision(2000, 4000)
direct_frequency_comparision(2700, 3700)
#direct_frequency_comparision(2500, 4500)

"""freq_mhz = [1e3 * f for f in freq if ((f * 1e3 > 0.5) & (f * 1e3 < 7.6))]
freq_mhz = [round(f, 6) for f in freq_mhz]
freq_mhz = np.unique(freq_mhz)
print(len(freq_mhz))
#print(freq_mhz)
print(freq_mhz)"""

#print(fft.fftfreq(data_calib1_pm1))

# https://stackoverflow.com/questions/57278821/how-does-one-fit-multiple-independent-and-overlapping-lorentzian-peaks-in-a-set
