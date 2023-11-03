# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:48:23 2022

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

def data_in_range(data, minf, maxf):
    return [f for f in data if (f*1e6 > minf) & (f*1e6 < maxf)]

N = 790560

freq = fftfreq(N, 20)[:N//2]

fft_data_calib1_pm1 = fft(data_calib1_pm1)
fft_data_calib1_pm2 = fft(data_calib1_pm2)

pm1 = data_in_range(freq, 2500, 4000)
pm2 = data_in_range(freq, 2500, 4000)

power = 2.0/N * np.abs(fft_data_calib1_pm1)[0:N//2]
power2 = 2.0/N * np.abs(fft_data_calib1_pm2)[0:N//2]

plt.plot(freq*1e6, power, color="black", linewidth=0.5)
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel (r'P ($\nu$)', size=14)
plt.xlim(3000, 3200)
plt.ylim(0.00, 0.15)
plt.show()

def add_peak(prefix, center, amplitude=1, sigma=3):
    peak = LorentzianModel(prefix=prefix)
    pars = peak.make_params()
    pars[prefix + 'center'].set(center)
    pars[prefix + 'amplitude'].set(amplitude, min=0, max=5)
    pars[prefix + 'sigma'].set(sigma, min=0)
    return peak, pars

#model = QuadraticModel(prefix='bkg_')
#params = model.make_params(a=1e-5, b=1e-6, c=0.01)
    
rorder = [ 3.,  4.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 50.,
 51., 52., 53., 54.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.,
 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44.,  5.,
  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35.,
 36., 37., 38., 39., 40., 41., 42., 43.,  4.,  5.,  6.,  7.,  8.,  9., 10.,
 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40.,
 41., 42., 43.]

degree = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2.,
 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
 2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
 3., 3., 3., 3., 3., 3.]

theoretical_frequencies = [0.535308, 0.679501, 0.970528, 1.115391, 1.260426, 1.403781, 1.543825,
 1.681478, 1.817298, 1.953574, 2.090097, 2.225231, 2.359319, 2.492708,
 2.626554, 2.761449, 2.896414, 3.031626, 3.16692 , 3.302485, 3.438826,
 3.575348, 3.712182, 3.849159, 3.986249, 4.123775, 4.261271, 4.398846,
 4.536387, 4.673751, 4.946911, 5.095484, 5.234301, 5.373123, 5.512118,
 5.65101 , 5.78991 , 5.928871, 6.067785, 6.206761, 6.345584, 6.48434 ,
 6.62311 , 6.761761, 7.038738, 7.176977, 7.315157, 7.453125, 7.590913,
 0.891473, 1.036782, 1.182289, 1.325785, 1.468123, 1.606647, 1.743139,
 1.879592, 2.015963, 2.152349, 2.28712 , 2.42074 , 2.554625, 2.689054,
 2.824228, 2.959557, 3.094791, 3.23037 , 3.366419, 3.502967, 3.639851,
 3.776819, 3.913997, 4.051451, 4.189064, 4.326765, 4.464404, 4.601967,
 4.739407, 4.876241, 5.023275, 5.162115, 5.301017, 5.44002 , 5.579044,
 5.718011, 5.857005, 5.996009, 6.135015, 6.273986, 0.957193, 1.101802,
 1.246902, 1.390052, 1.530353, 1.668165, 1.804061, 1.940584, 2.077293,
 2.212784, 2.34722 , 2.480858, 2.615009, 2.750181, 2.885445, 3.02098 ,
 3.156551, 3.292407, 3.42903 , 3.565824, 3.702942, 3.840181, 3.977531,
 4.115312, 4.253053, 4.390873, 4.528647, 4.66624 , 4.803671, 4.939921,
 5.088536, 5.227532, 5.366524, 5.505683, 5.644735, 5.783787, 5.922895,
 6.06195 , 6.201058, 0.864609, 1.012056, 1.158037, 1.302591, 1.445901,
 1.585199, 1.722534, 1.859199, 1.995954, 2.132971, 2.268246, 2.402443,
 2.536677, 2.671394, 2.807062, 2.942791, 3.078497, 3.214521, 3.350926,
 3.487938, 3.625222, 3.762608, 3.900194, 4.037984, 4.175992, 4.314043,
 4.452032, 4.589953, 4.727719, 4.864958, 4.999742, 5.151218, 5.290399,
 5.429645, 5.56893 , 5.708133, 5.847348, 5.986578, 6.125788, 6.264974]

#theoretical_frequencies = [0.535308, 0.679501, 0.970528, 1.115391, 1.260426, 1.403781, 1.543825, 1.681478, 1.817298, 1.953574, 2.090097, 2.225231, 2.359319, 2.492708, 2.626554, 2.761449, 2.896414, 3.031626, 3.16692, 3.302485, 3.438826, 3.575348, 3.712182, 3.849159, 3.986249, 4.123775, 4.261271, 4.398846, 4.536387, 4.673751, 4.946911, 5.095484, 5.234301, 5.373123, 5.512118, 5.65101, 5.78991, 5.928871, 6.067785, 6.206761, 6.345584, 6.48434, 6.62311, 6.761761, 7.038738, 7.176977, 7.315157, 7.453125, 7.590913, 0.891473, 1.036782, 1.182289, 1.325785, 1.468123, 1.606647, 1.743139, 1.879592, 2.015963, 2.152349, 2.28712, 2.42074, 2.554625, 2.689054, 2.824228, 2.959557, 3.094791, 3.23037, 3.366419, 3.502967, 3.639851, 3.776819, 3.913997, 4.051451, 4.189064, 4.326765, 4.464404, 4.601967, 4.739407, 4.876241, 5.023275, 5.162115, 5.301017, 5.44002, 5.579044, 5.718011, 5.857005, 5.996009, 6.135015, 6.273986, 0.957193, 1.101802, 1.246902, 1.390052, 1.530353, 1.668165, 1.804061, 1.940584, 2.077293, 2.212784, 2.34722, 2.480858, 2.615009, 2.750181, 2.885445, 3.02098, 3.156551, 3.292407, 3.42903, 3.565824, 3.702942, 3.840181, 3.977531, 4.115312, 4.253053, 4.390873, 4.528647, 4.66624, 4.803671, 4.939921, 5.088536, 5.227532, 5.366524, 5.505683, 5.644735, 5.783787, 5.922895, 6.06195, 6.201058, 0.864609, 1.012056, 1.158037, 1.302591, 1.445901, 1.585199, 1.722534, 1.859199, 1.995954, 2.132971, 2.268246, 2.402443, 2.536677, 2.671394, 2.807062, 2.942791, 3.078497, 3.214521, 3.350926, 3.487938, 3.625222, 3.762608, 3.900194, 4.037984, 4.175992, 4.314043, 4.452032, 4.589953, 4.727719, 4.864958, 4.999742, 5.151218, 5.290399, 5.429645, 5.56893, 5.708133, 5.847348, 5.986578, 6.125788, 6.264974]

def fit_peaks(fmin, fmax, pm=1, figsize='large'):
    
    if figsize == 'large':
        plt.figure(figsize=(16,8))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    
    within_range = np.argwhere((np.array(theoretical_frequencies)>fmin) & (np.array(theoretical_frequencies)<fmax))
    
    wr = []
    
    for element in within_range:
        wr.append(element[0])
    
    labels = []

    rough_peak_positions = (1e3* f for f in theoretical_frequencies if ((f > fmin) & (f < fmax)))
    
    """model = LinearModel()
    params = model.make_params()
    params['slope'].set(0, vary=False)
    params['intercept'].set(0, vary=False)"""
    
    model = QuadraticModel(prefix='bkg_')
    params = model.make_params(a=1e-7, b=1e-5, c=0.005)
    
    for i, cen in enumerate(rough_peak_positions):
        peak, pars = add_peak('lz%d_' % (i+1), cen)
        label = 'n = %d, l = %d' % (rorder[wr[i]], degree[wr[i]])
        labels.append(label)
        model = model + peak
        params.update(pars)
        
    print(labels)
        
    xdat = np.array(data_in_range(freq, fmin*1e3, fmax*1e3))*1e6
    if pm == 2:
        ydat = np.array(power2[(freq > fmin*1e-3) & (freq < fmax*1e-3)])
    else:
        ydat = np.array(power[(freq > fmin*1e-3) & (freq < fmax*1e-3)])
    
    print(len(xdat))
    print(len(ydat))
        
    var = np.var(ydat)
    
    #init = model.eval(params, x=xdat)
    result = model.fit(ydat, params, x=xdat, weights=np.sqrt(1.0/var))
    comps = result.eval_components()
    
    print(result.fit_report(min_correl=0.5))
    
    labels = np.insert(labels, 0, 'Background')
    
    plt.plot(xdat, ydat, label='Data', color="black", linewidth=0.5)
    plt.plot(xdat, result.best_fit, label='Best Fit', color ="darkred", linewidth=1.5)
    j = 0
    """for name, comp in comps.items():
        plt.plot(xdat, comp, '--', label=labels[j], linewidth=0.5)
        j += 1"""
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'P ($\nu$) (m$^2$s$^{-2}\mu$Hz$^{-1}$)', size=16)
    plt.legend(loc='upper right', prop={'size': 14})
    plt.show()
    
    fitted_frequencies = []
    fitted_frequencies_err = []
    fitted_frequencies_n = []
    fitted_frequencies_l = []
    fitted_linewidths = []
    fitted_linewidths_err = []
    
    for i in range (len(wr)):
        fitted_frequencies.append(result.params['lz%d_center' % (i+1)].value)
        fitted_frequencies_err.append(result.params['lz%d_center' % (i+1)].stderr)
        fitted_frequencies_n.append(rorder[wr[i]])
        fitted_frequencies_l.append(degree[wr[i]])
        fitted_linewidths.append(2*result.params['lz%d_sigma' % (i+1)].value)
        fitted_linewidths_err.append(result.params['lz%d_sigma' % (i+1)].stderr)
        
    print(fitted_frequencies)
    print(fitted_frequencies_err)
    print(fitted_frequencies_n)
    print(fitted_frequencies_l)
    print(fitted_linewidths)
    print(fitted_linewidths_err)
    print([1e3* f for f in theoretical_frequencies if ((f > fmin) & (f < fmax))])
#fit_peaks(3.0, 3.2)
#fit_peaks(2.7, 3.7)
#fit_peaks(2.7, 3.7, 2)
fit_peaks(3.7, 4)
#fit_peaks(3.7, 5, 2)
#fit_peaks(1, 5)