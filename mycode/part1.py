# %%
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

# %% [markdown]
# # Plot Sound Speed Profile

# %% [markdown]
# ## Read Files

# %%
ssp_basu_path = '../SoundSpeed/basu_09.txt'
column_names = [" r/R_sun", "c [cm/s]", "Sigma_c"]
ssp_obs = pd.read_csv(ssp_basu_path, delim_whitespace=True, skiprows=8, names=column_names)
#print(ssp_obs)

# %%
data_gs98_path = "../solar_23.05.1/solar.profile"
skiprows = 6
column_names = ["radius", "csound"]

model_gs98 = pd.read_csv(data_gs98_path, delim_whitespace=True, skiprows=skiprows, usecols=[9, 10], names=column_names)
#print(model_gs98)

# %% [markdown]
# ## Interpolate the observed data
# Since the observed data will be used to compare both models
# 
# (acho que em todos os fichs a velocidade do som esta em cm/s)

# %%
c_obs = interpolate.interp1d( x = ssp_obs[" r/R_sun"],
                             y = ssp_obs["c [cm/s]"], 
                             kind='cubic', 
                             bounds_error=False  # Allow extrapolation
                             )

x = np.linspace(model_gs98["radius"].min(), model_gs98["radius"].max(),100)
y = c_obs(x)

plt.plot( ssp_obs[" r/R_sun"], ssp_obs["c [cm/s]"],label = "obs points",marker="o", linestyle = 'None')
plt.plot(x,y,label='obs interpol cubic')
plt.plot(model_gs98["radius"],model_gs98["csound"], label = "gs98 model",marker=".",linestyle = 'None')
plt.legend(numpoints=1)
plt.show()


# %%



