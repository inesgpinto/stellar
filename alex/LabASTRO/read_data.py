# -*- coding: utf-8 -*-
"""
Alexandre Barbosa
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LinearModel 

def read_data(file):
    f = open(file, "r")
    lines = f.readlines()
    degree = []
    rorder = []
    eigenfreq = []
    for line in lines:
        line = line.split()
        if len(line) > 0:
            degree.append(float(line[0]))
            rorder.append(float(line[1]))
            eigenfreq.append(float(line[-1]))
    f.close()
    
    return [degree, rorder, eigenfreq]


data = read_data("specTCL93l03.txt")
data_2 = read_data("specTCL93.txt")

degree = np.array(data[0]) # l
rorder = np.array(data[1]) # n
eigenfreq = np.array(data[2]) # em mHz

"""degree_2 = data_2[0]
rorder_2 = data_2[1]
eigenfreq_2 = data_2[2]"""

"""print(eigenfreq)
print(min(eigenfreq))
print(max(eigenfreq))

print(degree)
print(rorder)"""

print(np.array2string(rorder, separator=', '))
print(np.array2string(degree, separator=', '))
print(np.array2string(eigenfreq, separator=', '))

def small_separation(l):
    delta = []
    frequency = []
    
    for n in rorder:
        delta.append(eigenfreq[(rorder == n) & (degree == l)] - eigenfreq[(rorder == n-1) & (degree == l + 2)] )
        if (delta[-1].size > 0):
            frequency.append(eigenfreq[(rorder == n) & (degree == l)])
        
    # unpack the nested list
    separation = []
    freq_nl = []

    for element in delta:
        if element.size > 0:
            separation.append(element[0])
            
    for f in frequency:
        if f.size > 0:
            freq_nl.append(f[0])
            
    #print(len(separation))
    #print(len(freq_nl))
        
    return [separation, freq_nl]

def large_separation(l):
    delta = []
    frequency = []
    
    for n in rorder:
        delta.append(eigenfreq[(rorder == n + 1) & (degree == l)] - eigenfreq[(rorder == n) & (degree == l)])
        if (delta[-1].size > 0):
            frequency.append(eigenfreq[(rorder == n) & (degree == l)])
        
    # unpack the nested list
    separation = []
    freq_nl = []
    
    for element in delta:
        if element.size > 0:
            separation.append(element[0])
            
    for f in frequency:
        if f.size > 0:
            freq_nl.append(f[0])
        
    return [separation, freq_nl]
        
# Small Separation: for l = 0 and l = 1

"""print(large_separation(0)[0])
print(large_separation(0)[1])
print(large_separation(1)[0])"""

# Large Separation: for l = 0, 1, 2, 3

"""print(small_separation(0))
print(small_separation(1))
print(small_separation(2))
print(small_separation(3))"""


"""plt.scatter(small_separation(0)[1], small_separation(0)[0] , color="black")
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', size=14)
plt.ylim(0.0050, 0.0150)
plt.show()

plt.scatter(small_separation(1)[1], small_separation(1)[0] , color="blue")
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel (r'$\delta \nu$ ($\mu$Hz)', size=14)
plt.ylim(0.0075, 0.0300)
plt.show()"""

colors = ['darkblue', 'darkcyan', 'darkgreen', 'olivedrab']
markers = ['d', 'x', 's', 'P']

plt.scatter([1e3 * x for x in small_separation(0)[1] ], [1e3 * x for x in small_separation(0)[0] ] , 
            color=colors[0], marker=markers[0], label= 'l = 0', s=19)
plt.scatter([1e3 * x for x in small_separation(1)[1] ], [1e3 * x for x in small_separation(1)[0] ], 
            color=colors[1], marker=markers[1], label= 'l = 1', s=19)
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel (r'$\delta \nu$ ($\mu$Hz)', size=14)
plt.ylim(5, 30)
plt.xlim(1000, 5000)
plt.legend()
plt.show()

for l in range(4):  
    plt.scatter([1e3 * x for x in large_separation(l)[1] ], [1e3 * x for x in large_separation(l)[0] ], 
                 marker=markers[l], s=19, color=colors[l], label= 'l = ' + str(l))
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'$\Delta \nu$ ($\mu$Hz)', size=14)
    plt.ylim(130, 148)
    plt.xlim(1000, 5000)
    
plt.legend(loc='upper right')
plt.show()

print(np.array2string(eigenfreq[degree == 0 & (rorder >= 19) & (rorder <= 24)], separator=', '))
print(np.array2string(eigenfreq[degree == 1 & (rorder >= 19) & (rorder <= 24)], separator=', '))
print(np.array2string(eigenfreq[degree == 2 & (rorder >= 19) & (rorder <= 24)], separator=', '))
print(np.array2string(eigenfreq[degree == 3 & (rorder >= 19) & (rorder <= 24)], separator=', '))
    

ff_1 = np.array([2764.1809602705034, 2898.838215384972, 3033.6094558011227, 3168.6494617287617, 3303.1642598896815, 3438.9140958966586, 3574.695821159006, 2828.072712397188, 2963.6455194917608, 3098.0155956602857, 3233.3687962754707, 3368.740947018557, 3503.6960261221493, 3640.1958866479777, 2754.619072364993, 2889.835722391421, 3024.4464118786314, 3159.7463386572226, 3295.11983040476, 3430.8763251105825, 3566.718677400696, 2811.33195117672, 2947.2076461141014, 3081.751943989268, 3218.11210273913, 3354.32686269819, 3490.0663932953585, 3626.5400139946137])
tff = np.array([2761.4489999999996, 2896.414, 3031.626, 3166.92, 3302.4849999999997, 3438.826, 3575.348, 2824.228, 2959.5570000000002, 3094.7909999999997, 3230.3700000000003, 3366.419, 3502.967, 3639.851, 2750.181, 2885.4449999999997, 3020.98, 3156.551, 3292.4069999999997, 3429.03, 3565.824, 2807.0620000000004, 2942.791, 3078.497, 3214.521, 3350.926, 3487.938, 3625.2219999999998])
nf = np.array([19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0])
lf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
err_f1 = np.array([0.06503345736333777, 0.03598979247664338, 0.011489964175166002, 0.02311086837044476, 0.03901053860421301, 0.06508133714786674, 0.15055624929947092, 0.04024689606248935, 0.030576491689771685, 0.03832036325722694, 0.035588276995514055, 0.043324618204416174, 0.07011047855804203, 0.13011167835080703, 0.11642141084612391, 0.061408801863009364, 0.056322976364785136, 0.08822171060911733, 0.0634959305894257, 0.07776339816414123, 0.21533910762803804, 0.29446458543656673, 0.22018823600380072, 0.1689586404094609, 0.21214979874581769, 0.3659273341560524, 0.5634944643547611, 0.8563994012630117])

ff_2 = np.array([2764.188087764187, 2898.8455590085246, 3033.6096758197614, 3168.646456317685, 3303.1637018037786, 3438.8940567868895, 3574.6822845902643, 2828.078727378834, 2963.653183598868, 3098.021461478759, 3233.3751660495186, 3368.741600631668, 3503.686520902472, 3640.1884692624394, 2754.6233100864274, 2889.8149090212078, 3024.4230750976185, 3159.7652322437007, 3295.1318232741846, 3430.8575055775023, 3566.756921684375, 2811.389471275055, 2947.1969207629086, 3081.8053190699898, 3218.06719908899, 3354.296621001131, 3489.970279056535, 3626.7331933512955])
err_f2 = np.array([0.06270752914487547, 0.03582158749571924, 0.011768870588282118, 0.022741257161391793, 0.03904388015447178, 0.06549717920830926, 0.15211020825477992, 0.04171501834936629, 0.030844661949449136, 0.03841230359168821, 0.03531424241606401, 0.04261164199496128, 0.06919595060054196, 0.13300286212113618, 0.11732184868599219, 0.06210055729278855, 0.05603434592848503, 0.08609041521873814, 0.06284322200328646, 0.07771536637459484, 0.2121567697260291, 0.2918366568855329, 0.21932067355916554, 0.16898603824103514, 0.21455198096819497, 0.36658170397870665, 0.6007727251697039, 0.8973517712121338])

diff_f1 = (ff_1 - tff)/tff*100
diff_f2 = (ff_2 - tff)/tff*100

diffs = np.concatenate((diff_f1, diff_f2))
tff2 = np.concatenate((tff, tff))

err_dif1 = err_f1/tff*100
err_dif2 = err_f1/tff*100

lw_1 = np.array([2.170043588999471, 1.6465714364090132, 0.6297592597735107, 1.6689047083554271, 2.360387419513815, 3.2603173413104702, 4.580655544082928, 2.332533006251275, 2.3424800703303417, 3.021682931960224, 3.123726822018389, 3.1709847817143135, 4.49116909165864, 6.4503406465747215, 3.6140963486232245, 2.669292473419751, 3.439709599342904, 4.251486391033279, 3.720560685747186, 4.099737109992664, 7.0218792024555885, 3.7533954805506777, 4.04193340879168, 3.3249436147090536, 4.433256377665477, 5.822313800803837, 6.8190426739984105, 8.16182277289307])
lw_1_err = np.array([0.09545878481974664, 0.05180008278997016, 0.016354988974503847, 0.03382171471401207, 0.05836472634653561, 0.10073165573927313, 0.25472482963694376, 0.05786365194650428, 0.04381671020475272, 0.05519454415891474, 0.05160461439265915, 0.06347583234159886, 0.10678260405179052, 0.2105227522335168, 0.17423653661060481, 0.08895976254876384, 0.08135748333670738, 0.13137213560424466, 0.09587344363603964, 0.12072460671210772, 0.344109738346632, 0.4284507199076252, 0.3179242751785968, 0.24364147895496516, 0.3099033046044682, 0.5439292516898476, 0.867388509492547, 1.3868752779585294])

lw_2 = np.array([2.1349932575531216, 1.6230446703340968, 0.6446234917774287, 1.6482215289219857, 2.3506222257647753, 3.280734256094127, 4.529009519600684, 2.384755164170757, 2.342915756693939, 3.029313699195103, 3.09222688302291, 3.1336313946337286, 4.418216320054681, 6.469963308999654, 3.5382857954245104, 2.7249098229319113, 3.382888623394525, 4.155915219322106, 3.689853381816521, 4.081855613100242, 6.918793411149251, 3.76891509875621, 3.986712855177937, 3.35634723908419, 4.488158325945676, 5.714544558448381, 7.194594715659832, 8.428830469360932])
lw_2_err = np.array([0.0919148058959709, 0.05155394590172439, 0.01675120137493914, 0.03323416283097784, 0.058377308509447265, 0.10139569838882884, 0.25704714019800023, 0.06000027153007008, 0.0441954815254366, 0.05533619831238632, 0.05119442522793763, 0.06234363434038259, 0.10559448217577184, 0.2165166789515358, 0.17524289261175058, 0.0899961863168971, 0.08090765920770586, 0.12795728063621123, 0.09481814767194217, 0.12065584957719562, 0.33947502375461375, 0.4247106161005905, 0.31654899981297996, 0.24375191558754317, 0.3134826840368783, 0.5439219044196725, 0.9282692599527124, 1.4615788422998626])

"""model = LinearModel()
params = model.make_params()
result = model.fit(diffs, params, x=tff2)
print(result.fit_report())
plt.plot(tff2, result.best_fit, label='Best Fit', color ="darkslategray", ls="dashed")"""
#plt.scatter(tff, diff_f1,  color=colors[0], marker=markers[0], label= 'PM 1', s=19)
#plt.scatter(tff, diff_f2, color=colors[1], marker=markers[1], label= 'PM 2', s=19)
plt.errorbar(tff, diff_f1, err_dif1,  color=colors[0], marker=markers[0], label= 'PM 1', fmt="o" )
plt.errorbar(tff, diff_f2, err_dif2, color=colors[1], marker=markers[1], label= 'PM 2', fmt="o")
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel (r'$\epsilon_\nu$ ($\%$)', size=14)
#plt.ylim(-2, 6)
plt.xlim(2700, 3700)
plt.legend()
plt.show()

freq_echelle0 = tff[lf == 0] % 135.3
freq_echelle1 = tff[lf == 1] % 135.3
freq_echelle2 = tff[lf == 2] % 135.3
freq_echelle3 = tff[lf == 3] % 135.3

plt.scatter(freq_echelle0, ff_1[lf == 0],  color=colors[0], marker=markers[0], s=19)
plt.scatter(freq_echelle1, ff_1[lf == 1],  color=colors[1], marker=markers[1], s=19)
plt.scatter(freq_echelle2, ff_1[lf == 2],  color=colors[2], marker=markers[2], s=19)
plt.scatter(freq_echelle3, ff_1[lf == 3],  color=colors[3], marker=markers[3], s=19)
plt.plot(freq_echelle0, ff_1[lf == 0],  color=colors[0], marker=markers[0], label= 'l=0',ls='--', linewidth='1.5')
plt.plot(freq_echelle1, ff_1[lf == 1],  color=colors[1], marker=markers[1], label= 'l=1', ls='--', linewidth='1.5')
plt.plot(freq_echelle2, ff_1[lf == 2],  color=colors[2], marker=markers[2], label= 'l=2', ls='--', linewidth='1.5')
plt.plot(freq_echelle3, ff_1[lf == 3],  color=colors[3], marker=markers[3], label= 'l=3', ls='--', linewidth='1.5')
plt.ylabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.xlabel (r'$\nu$ mod. <$\Delta_{\nu}$> ($\mu$Hz)', size=14)
plt.xlim(0, 135.3)
plt.legend(loc='best')
plt.show()


#err_sigma_1 = (ff_1 - tff)/err_f1
#err_sigma_2 = (ff_2 - tff)/err_f2

#print(err_sigma_1)
#print(err_sigma_2)

ff_10 = (ff_1[lf == 0]-tff[lf == 0])/tff[lf == 0]*100
ff_11 = (ff_1[lf == 1]-tff[lf == 1])/tff[lf == 1]*100
ff_12 = (ff_1[lf == 2]-tff[lf == 2])/tff[lf == 2]*100
ff_13 = (ff_1[lf == 3]-tff[lf == 3])/tff[lf == 3]*100

ff_20 = (ff_2[lf == 0]-tff[lf == 0])/tff[lf == 0]*100
ff_21 = (ff_2[lf == 1]-tff[lf == 1])/tff[lf == 1]*100
ff_22 = (ff_2[lf == 2]-tff[lf == 2])/tff[lf == 2]*100
ff_23 = (ff_2[lf == 3]-tff[lf == 3])/tff[lf == 3]*100

#transform into errorbars
plt.scatter(tff[lf == 0], ff_10,  color=colors[0], marker=markers[0], label= 'l=0', s=19)
plt.scatter(tff[lf == 1], ff_11, color=colors[1], marker=markers[1], label= 'l=1', s=19)
plt.scatter(tff[lf == 2], ff_12,  color=colors[2], marker=markers[2], label= 'l=2', s=19)
plt.scatter(tff[lf == 3], ff_13, color=colors[3], marker=markers[3], label= 'l=3', s=19)
"""plt.errorbar(tff[lf == 0], ff_10,  yerr=err_f1[lf == 0], color=colors[0], marker=markers[0], label= 'l=0')
plt.errorbar(tff[lf == 1], ff_11, yerr=err_f1[lf == 1], color=colors[1], marker=markers[1], label= 'l=1')
plt.errorbar(tff[lf == 2], ff_12,  yerr=err_f1[lf == 2], color=colors[2], marker=markers[2], label= 'l=2')
plt.errorbar(tff[lf == 3], ff_13, yerr=err_f1[lf == 3], color=colors[3], marker=markers[3], label= 'l=3')"""

plt.scatter(tff[lf == 0], ff_20,  color=colors[0], marker=markers[0], s=19)
plt.scatter(tff[lf == 1], ff_21, color=colors[1], marker=markers[1], s=19)
plt.scatter(tff[lf == 2], ff_22,  color=colors[2], marker=markers[2], s=19)
plt.scatter(tff[lf == 3], ff_23, color=colors[3], marker=markers[3], s=19)
plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
plt.ylabel (r'$\epsilon_\nu$ ($\%$)', size=14)
#plt.ylim(-2, 6)
plt.xlim(2700, 3700)
plt.legend()
plt.show()

print(ff_10)

""" Dúvidas: 1) Como é que se compara diretamente as frequências?
    
    2) Como é que se escolhe o sampling rate? """
    
def small_separation_exp(ff, err, l):
    delta = []
    frequency = []
    err_frequency = []
    
    for n in rorder:
        delta.append(ff[(nf == n) & (lf == l)] - ff[(nf == n-1) & (lf == l + 2)] )
        err_frequency.append(err[(nf == n) & (lf == l)] + err[(nf == n-1) & (lf == l + 2)] )
        if (delta[-1].size > 0):
            frequency.append(ff[(nf == n) & (lf == l)])
        
    # unpack the nested list
    separation = []
    freq_nl = []
    err_f = []

    for element in delta:
        if element.size > 0:
            separation.append(element[0])
            
    for f in frequency:
        if f.size > 0:
            freq_nl.append(f[0])
            
    for f in err_frequency:
        if f.size > 0:
            err_f.append(f[0])
            
    #print(len(separation))
    #print(len(freq_nl))
        
    return [separation, freq_nl, err_f]

def large_separation_exp(ff, err, l):
    delta = []
    frequency = []
    err_frequency = []
    
    for n in rorder:
        delta.append(ff[(nf == n + 1) & (lf == l)] - ff[(nf == n) & (lf == l)])
        err_frequency.append(err[(nf == n + 1) & (lf == l)] + err[(nf == n) & (lf == l)] )
        if (delta[-1].size > 0):
            frequency.append(ff[(nf == n) & (lf == l)])
        
    # unpack the nested list
    separation = []
    freq_nl = []
    err_f = []
    
    for element in delta:
        if element.size > 0:
            separation.append(element[0])
            
    for f in frequency:
        if f.size > 0:
            freq_nl.append(f[0])
            
    for f in err_frequency:
        if f.size > 0:
            err_f.append(f[0])
        
    return [separation, freq_nl, err_f]

for l in range(4):  
    plt.errorbar([x for x in large_separation_exp(ff_1, err_f1, l)[1] ], [x for x in large_separation_exp(ff_1, err_f1, l)[0] ], yerr=[x for x in large_separation_exp(ff_1, err_f1, l)[2] ],
                 marker=markers[l], fmt='o', color=colors[l], label= 'l = ' + str(l))
    plt.plot([1e3 * x for x in large_separation(l)[1] ], [1e3 * x for x in large_separation(l)[0] ], ls='-.', lw=1.2, alpha=0.8, color=colors[l])
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'$\Delta \nu$ ($\mu$Hz)', size=14)
    plt.ylim(134, 138)
    plt.xlim(2700, 3550)
    
plt.legend(loc='upper left')
plt.show()

plt.show()

for l in range(2):  
    mag = 5
    if l == 1:
        mag = 3
    plt.errorbar([x for x in small_separation_exp(ff_1, err_f1, l)[1] ], [x for x in small_separation_exp(ff_1, err_f1, l)[0] ], yerr= [mag*x for x in small_separation_exp(ff_1, err_f1, l)[2]], 
                 marker=markers[l], fmt='o', color=colors[l], label= 'l = ' + str(l))
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'$\delta \nu$ ($\mu$Hz)', size=14)
    plt.ylim(5, 20)
    plt.xlim(2800, 3650)

plt.legend()
    
#plt.plot([1e3 * x for x in small_separation(0)[1] ], [1e3 * x for x in small_separation(0)[0] ] , 
            #color=colors[0], ls='--', lw=1)
#plt.plot([1e3 * x for x in small_separation(1)[1] ], [1e3 * x for x in small_separation(1)[0] ], 
            #color=colors[1], linestyle='--', lw=1)
model0 = LinearModel()
params = model0.make_params()
result0 = model0.fit([x for x in small_separation_exp(ff_1, err_f1, 0)[0] ], params, x=[x for x in small_separation_exp(ff_1, err_f1, 0)[1] ])
print(result0.fit_report())
plt.plot([x for x in small_separation_exp(ff_1, err_f1, l)[1] ], result0.best_fit, color =colors[0], ls="-.", lw=1)


model1 = LinearModel()
params = model1.make_params()
result1 = model1.fit([x for x in small_separation_exp(ff_1, err_f1, 1)[0] ], params, x=[x for x in small_separation_exp(ff_1, err_f1, 1)[1] ])
print(result1.fit_report())
plt.plot([x for x in small_separation_exp(ff_1, err_f1, l)[1] ], result1.best_fit, color =colors[1], ls="-.", lw=1)


#plt.legend()
plt.show()

plt.errorbar([x for x in large_separation_exp(ff_1, err_f1, l)[1] ], [x for x in large_separation_exp(ff_1, err_f1, l)[0] ], yerr=[x for x in large_separation_exp(ff_1, err_f1, l)[2] ],
                 marker=markers[l], fmt='o', color=colors[l], label= 'l = ' + str(l))

avg_ff = []

for i in range (len(ff_1)):
    avg_ff.append(0.5*(ff_1[i] + ff_2[i]))
    
def printstuff(l):
    
    err_sigma = (ff_1[lf == l] - tff[lf == l])/lw_1[lf == l]
    
    #for i in range(7):
        
        #print({round(ff_1[lf == l][i],2)} + "( " + {round(err_f1[lf == l][i],2)} + ")" + " & " + {round(tff[lf == l])[i], 2)} + " & " + " & " + {round(err_sigma[i], 2)} + " \\ ")
              
              
        #print("\n")

    for item in (ff_1[lf == l]):
        print(round(item, 2), " & " )
        
    for item in (err_f1[lf == l]):
        print(round(item, 2), " & " )
        
    for item in (tff[lf == l]):
        print(round(item, 2), " & " )
        
    if l == 0:
        for item in ff_10:
             print(round(item, 2), " & " )
    if l == 1:
        for item in ff_11:
             print(round(item, 2), " & " )
             
    if l == 2:
        for item in ff_12:
             print(round(item, 2), " & " )
             
    if l == 3:
        for item in ff_13:
             print(round(item, 2), " & " )
         
    err_sigma = (ff_1[lf == l] - tff[lf == l])/lw_1[lf == l]
    
    for item in err_sigma:
        print(round(item, 2))
        
#printstuff(3)

#print(np.average([x for x in small_separation_exp(ff_1, err_f1, l)[0] ] ))
#print(np.average([x for x in large_separation_exp(ff_1, err_f1, l)[0] ] ))

lw_h = np.array([9.888770556934062, 9.312536684027295, 12.23111697655066, 17.249820791185638, 15.147925872329076, 11.431788150907563, 10.32489134104822, 1.9450734865047399, 36.11023951117836, 6.133233423560414, 0.10646689319772662, 15.052166011726921, 19.777846048846893, 16.705302740872654, 18.91694321538862, 17.695391704701812, 26.302052971406635, 39.917463404327854, 6.719965652593528, 11.430180223536048, 16.518754092632726, 18.926069789086352, 15.44473269107802, 21.214705582488627, 23.677134623861722, 31.925449888572096, 28.57257537446742, 12.002637914441, 24.49647473905561, 15.042035732364333, 9.617141753953838, 0.010079418163214893, 19.10193026818115, 10.406603607827387, 18.34069582820422, 6.78699809740511, 0.13067494865454155, 0.7606266041439627])
lw_h_err = np.array([0.3656170526412042, 0.4860652031447027, 0.7928307267077472, 3.5093346877183627, 2.052419450414115, 4.117525281549205, 2.7923117403983957, 0.7689405046579089, 17.251417834543048, 0.3227353989449977, 0.013629508101000604, 0.48429323288185383, 0.5932462521567922, 1.2845457783775018, 1.3234683862424246, 2.739587491211576, 3.1603620813575004, 5.9872752501290485, 0.6029762583729957, 0.6023765064096136, 2.0163208539137067, 3.439350356493558, 1.4246605473145348, 1.6398550892972157, 2.584351188635391, 2.5551489706568575, 3.271552374751256, 21201927.33672852, 1.400146725056908, 0.3199446966652399, 2.338719631519615, 0.2905587911216286, 3.7251240849014247, 4.821136404046495, 3.850221035617529, 2.881285362636226, 0.04161434216397619, 154.32244750798304])

lw_l = np.array([2.336350300045426, 0.24775033274573266, 0.437518847669077, 0.4142457977204006, 0.2404893674733053, 1.0219394664944774, 1.5133305003769766, 1.645602657041744, 2.045356744744983, 0.029457726727513567, 0.6370124626171965, 0.5815383717733793, 1.7045860037993008, 1.6779551208571215, 1.7421285841797252, 2.062812820598114, 2.2697098000016576, 2.338380503609632, 3.2928464863077878, 0.21645050629248574, 1.772898181408853, 1.8320564147404106, 2.0255904403064706, 2.0759343949713065, 2.2537623778857627, 3.22625786976145, 3.579529289190333, 0.24515372342174047, 1.8879770241812914, 1.102837174902397, 17.032813213104728, 2.509569566709879, 1.8793154956987603, 2.1111388459041827, 4.351282651096822, 3.7970899553468884])
lw_l_err = np.array([1.6642589230108098, 0.04797083761413172, 0.056341080828133376, 0.04751527079526796, 0.0146105911359504, 0.05119023777718898, 0.06276695484259766, 0.04355120383526079, 0.03767797220025836, 0.07119810878988378, 0.09921501016394155, 0.0588813652543059, 0.12480860321772035, 0.07898465844918194, 0.05591826864121375, 0.042814869300291034, 0.0331306105638811, 0.022179558210895307, 2.013683159526543, 0.08978315130456743, 0.39114745101284576, 0.2432758876950956, 0.21021151156149462, 0.12215191440930363, 0.0782840068175324, 0.08728605699853752, 0.07736518574576458, 0.14329851778702202, 1.0662945491919233, 0.7434609966131476, 8.212655845185175, 0.5133922902404986, 0.30774684891386006, 0.2070215827182878, 0.36520759911165857, 0.2164209540681597])

llow = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
lhigh = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

tflow = np.array([1543.825, 1681.478, 1817.298, 1953.5739999999998, 2090.097, 2225.2309999999998, 2359.319, 2492.708, 2626.554, 1606.647, 1743.139, 1879.5919999999999, 2015.9630000000002, 2152.349, 2287.12, 2420.74, 2554.625, 2689.054, 1530.353, 1668.165, 1804.061, 1940.584, 2077.293, 2212.784, 2347.2200000000003, 2480.858, 2615.009, 1585.199, 1722.534, 1859.199, 1995.954, 2132.971, 2268.246, 2402.4429999999998, 2536.677, 2671.394])
tfhigh = np.array([3712.182, 3849.159, 3986.249, 4123.775000000001, 4261.271, 4398.846, 4536.387000000001, 4673.751, 4946.911, 3776.819, 3913.9970000000003, 4051.451, 4189.064, 4326.765, 4464.404, 4601.967000000001, 4739.407, 4876.241, 3702.942, 3840.181, 3977.531, 4115.312, 4253.053000000001, 4390.873, 4528.647, 4666.24, 4803.670999999999, 4939.921, 3762.608, 3900.194, 4037.984, 4175.992, 4314.043, 4452.032, 4589.953, 4727.718999999999, 4864.958, 4999.742])

fflow = np.array([1553.687595088596, 1686.501822952768, 1822.1875434238432, 1957.4341691442967, 2093.5994057730973, 2228.808341504094, 2362.7706397355737, 2496.0745251315575, 2629.82801531494, 1612.303676247866, 1749.688104989812, 1885.5529877194083, 2020.904675138178, 2156.781920122062, 2291.910222685985, 2425.558773394934, 2559.273673142343, 2693.315520076822, 1536.3508495513038, 1667.5989065189428, 1810.2743472981967, 1945.6542940503089, 2082.0252227232504, 2217.925290430482, 2352.123753548208, 2485.881005565187, 2619.5857355013477, 1592.2215438739374, 1719.258515588341, 1865.9740886211641, 1993.2774377157923, 2138.000470661117, 2272.270671944254, 2406.8848408465033, 2541.605698151036, 2675.917255266959])
fflow_err = np.array([1.151169113535333, 0.03370224334398632, 0.03977718091580696, 0.033525494940329074, 0.010280085862601094, 0.03601050027177342, 0.0440015988581315, 0.03028037764364521, 0.025800001833500605, 0.006605357198253944, 0.07000205750355343, 0.04155169603473374, 0.08686640855916257, 0.05545065908716811, 0.039285805691668, 0.030028416609054455, 0.023020188817730734, 0.015184319510575521, 1.3783046586189007, 0.06333189253881076, 0.27492851733081863, 0.17083019948364916, 0.14740377449015454, 0.08554221734880563, 0.05471575664629705, 0.06016966172553772, 0.05216165467699802, 0.10201724333487676, 0.7456227689998658, 0.5237425089766351, 5.40223596474518, 0.3591812241132026, 0.21607025068474794, 0.14522209508683678, 0.2509404684558346, 0.14569222280295088])

ffhigh = np.array([3710.2402465721398, 3847.5166864846806, 3983.7965038026737, 4119.949823167903, 4260.868824361591, 4404.0419934830215, 4539.034970615798, 4678.6800668243595, 4950.142041988111, 3776.638714630218, 3914.081657223286, 4049.72460001725, 4186.227220060333, 4326.901281566102, 4462.529578027482, 4603.592017940772, 4739.319850459406, 4876.68663568831, 3701.900527955793, 3838.3151898589554, 3974.284609343359, 4114.2139582101745, 4249.615575458463, 4390.132507973859, 4526.117411717775, 4666.910445080487, 4806.474684263177, 4949.576406096217, 3771.0130802172625, 3912.6713869524674, 4033.365657074827, 4172.952682669006, 4312.793759986334, 4448.784856901178, 4590.535814274313, 4721.171091834241, 4876.689052650835, 5003.52295165455])
ffhigh_err = np.array([0.257590532508899, 0.3233305965255321, 0.5871978084829061, 5.5769403255178815, 1.6613712373142269, 2.2614115384581375, 1.3225225124270814, 0.4571371817758409, 3.6149249076960817, 0.12512544184867846, 0.007394664396163847, 0.2874426968669423, 0.3349533472483566, 0.9284092243550867, 0.9438429892548652, 2.218328700600853, 1.628018603040565, 2.314464505324108, 0.2094614109265903, 0.4524146103122053, 2.2064781704853984, 9.358451447631467, 1.1965139979042372, 1.2885614904035338, 2.2990421116465707, 1.401539780033995, 1.3958116839089059, 9415800.541792776, 1.3186272594440525, 0.1878833289193119, 1.3578005059316365, 0.024355055372515057, 2.9589280766039594, 2.6589816526426744, 3.2268847721786633, 1.5213349215987808, 0.026187976508597714, 24.926641409226914])

nlow = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
nhigh = np.array([26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 35.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0])

#print(lw_l[llow == 0])
#print(lw_1[lf == 0])
#print(lw_h[lhigh == 0])

def lw(l):
    aux_arr =  np.concatenate((lw_l[llow == l], lw_1[lf == l]))
    aux_arr2 = np.concatenate((tflow[llow == l], tff[lf == l]))
    aux_arr3 = np.concatenate((lw_l_err[llow == l], lw_1_err[lf == l]))
    return [np.concatenate((aux_arr2, tfhigh[lhigh == l])) , np.concatenate((aux_arr, lw_h[lhigh == l])), np.concatenate((aux_arr3, lw_h_err[lhigh == l]))  ]

#lw1 = np.concatenate((lw_l[llow == 1], lw_1[lf == 1], lw_h[lhigh == 1]))
#lw2 = np.concatenate((lw_l[llow == 2], lw_1[lf == 2], lw_h[lhigh == 2]))
#lw3 = np.concatenate((lw_l[llow == 3], lw_1[lf == 3], lw_h[lhigh == 3]))

#allf1 = np.concatenate(tflow[llow == 1], tff[lf == 1], tfhigh[lhigh == 1])
#allf2 = np.concatenate(tflow[llow == 2], tff[lf == 2], tfhigh[lhigh == 2])
#allf3 = np.concatenate(tflow[llow == 3], tff[lf == 3], tfhigh[lhigh == 3])
#plt.show()

for l in range(4):
    plt.errorbar(lw(l)[0], lw(l)[1]*1000, yerr=lw(l)[2]*1000,  marker=markers[l], fmt='o', color=colors[l], label= 'l = ' + str(l), linestyle='--')
plt.yscale('log')
plt.ylabel(r'Linewidth (nHz)', size=14)
plt.xlabel (r'$\nu$ ($\mu$Hz)', size=14)
plt.ylim(100, 30000)
plt.xlim(1400, 4000)
plt.legend(loc='lower right')
plt.show()


def ss(l):
    aux_arr =  np.concatenate((fflow[llow == l], ff_1[lf == l]))
    aux_arr2 = np.concatenate((fflow_err[llow == l], err_f1[lf == l]))
    aux_arr3 = np.concatenate((nlow[llow == l], nf[lf == l]))
    aux_arr4 = np.concatenate((llow[llow == l], lf[lf == l]))
    aux_arr5 = np.concatenate((llow, lf))
    aux_arr6 = np.concatenate((nlow, nf))
    aux_arr7 = np.concatenate((tflow[llow == l], tff[lf == l]))
    return [np.concatenate((aux_arr, ffhigh[lhigh == l])) , np.concatenate((aux_arr2, ffhigh_err[lhigh == l])), np.concatenate((aux_arr3, nhigh[lhigh == l])), np.concatenate((aux_arr4, lhigh[lhigh == l])), np.concatenate((aux_arr5, lhigh)), np.concatenate((aux_arr6, nhigh)), np.concatenate((aux_arr7, tfhigh[lhigh == l]))]

def small_separation_exp2(ff, err, nf2, lf2, l):
    delta = []
    frequency = []
    err_frequency = []
    
    for n in nf2:
        delta.append(ff[(nf2 == n) & (lf2 == l)] - ff[(nf2 == n-1) & (lf2 == l + 2)] )
        err_frequency.append(err[(nf2 == n) & (lf2 == l)] + err[(nf2 == n-1) & (lf2 == l + 2)] )
        if (delta[-1].size > 0):
            frequency.append(ff[(nf2 == n) & (lf2 == l)])
        
    # unpack the nested list
    separation = []
    freq_nl = []
    err_f = []

    for element in delta:
        if element.size > 0:
            separation.append(element[0])
            
    for f in frequency:
        if f.size > 0:
            freq_nl.append(f[0])
            
    for f in err_frequency:
        if f.size > 0:
            err_f.append(f[0])
            
    print(len(separation))
    #print(len(freq_nl))
        
    return [separation, freq_nl, err_f]

def large_separation_exp2(ff, err, nf2, lf2, l):
    delta = []
    frequency = []
    err_frequency = []
    
    for n in nf2:
        delta.append(ff[(nf2 == n + 1) & (lf2 == l)] - ff[(nf2 == n) & (lf2 == l)])
        err_frequency.append(err[(nf2 == n + 1) & (lf2 == l)] + err[(nf2 == n) & (lf2 == l)] )
        if (delta[-1].size > 0):
            frequency.append(ff[(nf2 == n) & (lf2 == l)])
            
        
    # unpack the nested list
    separation = []
    freq_nl = []
    err_f = []
    
    for element in delta:
        if element.size > 0:
            separation.append(element[0])
            
    for f in frequency:
        if f.size > 0:
            freq_nl.append(f[0])
            
    for f in err_frequency:
        if f.size > 0:
            err_f.append(f[0])
            
    #print(separation)
    #print(freq_nl)
        
    return [separation, freq_nl, err_f]


for l in range(4):  
    plt.errorbar( large_separation_exp2(ss(l)[0], ss(l)[1], ss(l)[2], ss(l)[3], l)[1], [x for x in large_separation_exp2(ss(l)[0], ss(l)[1], ss(l)[2], ss(l)[3], l)[0] ], yerr= [x for x in large_separation_exp2(ss(l)[0], ss(l)[1], ss(l)[2], ss(l)[3], l)[2] ], 
                 marker=markers[l], fmt='o', color=colors[l], label= 'l = ' + str(l))
    plt.plot([1e3 * x for x in large_separation(l)[1] ], [1e3 * x for x in large_separation(l)[0] ], ls='-.', lw=1.2, alpha=0.8, color=colors[l])
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'$\Delta \nu$ ($\mu$Hz)', size=14)
    plt.ylim(130, 140)
    plt.xlim(1500, 4000)
    
plt.legend(loc='lower right')

plt.show()



for l in range(4):
    relerr = (np.array(ss(l)[0]) - np.array(ss(l)[6]))/ np.array(ss(l)[6])
    errerr = np.array(ss(l)[1]) / np.array(ss(l)[6])
    if (l == 0) or (l == 1) :
        plt.errorbar( ss(l)[6], relerr, yerr=errerr,
                 marker=markers[l], fmt='--o', color=colors[l], label= 'l = ' + str(l))
    else:
        plt.errorbar( ss(l)[6], relerr, yerr=errerr,
                 marker=markers[l], fmt='o', color=colors[l], label= 'l = ' + str(l))
    #plt.plot([1e3 * x for x in large_separation(l)[1] ], [1e3 * x for x in large_separation(l)[0] ], ls='-.', lw=1.2, alpha=0.8, color=colors[l])
    plt.xlabel(r'$\nu$ ($\mu$Hz)', size=14)
    plt.ylabel (r'$\delta \nu$ / $\nu$ ', size=14)
    plt.ylim(-0.003, 0.005)
    plt.xlim(1500, 4000)
    plt.legend(loc='upper right')
plt.show()

ssavg0 = np.average(small_separation_exp(ff_1, err_f1, 0)[0])
varssavg0 = np.var(small_separation_exp(ff_1, err_f1, 0)[0])

#ssavg1 = np.average(small_separation_exp(ff_1, err_f1, 1)[0])

lsavg = np.zeros(4)
lsvar = np.zeros(4)

l = 0

for l in range(4):
    lsavg[l] = np.average(large_separation_exp(ff_1, err_f1, l)[0])
    lsvar[l] = np.var(large_separation_exp(ff_1, err_f1, l)[0])


print(ssavg0)
print(varssavg0)

print(np.average(lsavg))
print(np.linalg.norm(lsvar))