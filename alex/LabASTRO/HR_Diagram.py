# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:46:49 2022

@author: alexa
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread("Astroseismic HR Plot.png")
#pts = np.array([[330,620],[950,620],[692,450],[587,450]])

hpx = 762
vpx = 541

xe = 0.66/250. * hpx  * 3
ye = 0.35/20. * vpx * 3

x = 135.41/250. * hpx 
y = 8.67/20. * hpx

fig = plt.imshow(image)
plt.errorbar(x, y, yerr=ye, xerr=xe,  color='darkorange', linewidth=1, marker='*', ms=5, capsize=3, label= 'Sun', fmt="o" )
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('pict.png', bbox_inches='tight', pad_inches = 0, dpi=1000)
plt.show()