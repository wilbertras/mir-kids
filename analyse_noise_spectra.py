"""
analyse_noise_spectra.py
This script analyzes and plot the noise spectra for all the relevant measurement configurations

"""


#--------------------------------------------------
# Load modules
# -------------------------------------------------
import matplotlib.pyplot as plt
import functions as ft
from functions_meta import get_noise_psd  
import matplotlibcolors
plt.style.use('matplotlibrc')
import numpy as np
import pickle
import re


#--------------------------------------------------
# Load kid_dict
# -------------------------------------------------

with open('path2data.txt', 'r') as file:
    path2data = file.readlines()[0]
path2kid_dict = r'%skid_dict.pkl' % path2data
with open(path2kid_dict, 'rb') as f:
    kid_dict = pickle.load(f)
print("Loaded file %s" % path2kid_dict)


#--------------------------------------------------
# Set general parameters for analysis
# -------------------------------------------------
name = 'KID26'
pw = 2000000                        # pulsewindow in microseconds
filetype = 'med'                    # filetype used for noise analysis: 'med' is sampled at 50kHz and 'vis' at 1MHz
filter = None                       # filter to be used for pulse rejection, None if no filter is used. We do not do pulse rejection for the most consistent comparison between all setups.
exclude_dc = True                   # exclude DC value


#--------------------------------------------------
# Analysis
# -------------------------------------------------

keys = ['BF dark', '3.8um off', '3.8um', '8.5um', '18.5um off', '18.5um', 'ADR dark', '25um off', '25um']
for key in keys:
    kid = kid_dict[name][key]['kid']
    pread = kid_dict[name][key]['pread']
    path = kid_dict[name][key]['dir']
    freqs, nxx = get_noise_psd(path, kid, pread, pw, filetype=filetype, filter=filter)
    kid_dict[name][key]['noise'] = [freqs, nxx]

with open(path2kid_dict, 'wb') as f:
    pickle.dump(kid_dict, f)
print("Updated file %s" % path2kid_dict)

#--------------------------------------------------
# Plotting
# -------------------------------------------------
keys = ['BF dark', None, None, '3.8um off', '3.8um', None, '8.5um', '18.5um off', '18.5um', 'ADR dark', '25um off', '25um']
labels = ['dark DR', ' ', ' ', '3.8 µm; $T_\mathrm{lab}=293\ K$', '3.8 µm; $T_\mathrm{lab}=293\ K$, QTH', ' ', '8.5 µm; $T_\mathrm{lab}=293\ K$', 
          '18.5 µm; $T_{bb}=3\ K$', '18.5 µm; $T_{bb}=160\ K$', 'dark ADR', '25 $\mu$m, $T_{bb}=3\ K$', '25 $\mu$m, $T_{bb}=24\ K$']
colors = ['k','None','None', 'b', 'b','None', 'y', 'o', 'o', 'k', 'p', 'p']
linestyles = ['-', ' ', ' ', '-', '--', ' ', '--', '-', '--', '-', '-', '--']

fig, axes = plt.subplot_mosaic('a;b', figsize=(18.5/1.85/2.54, 9/2.54), sharex=False, sharey=True, constrained_layout=True)
ylabel_psd = '$S_{\\theta}/(4Q_l)^2$ [dBc/Hz]'
xlabel_psd = 'Frequency [Hz]'
ylim_psd = [-175, -145]
xlim_psd = [10, 2e4]
xticks = np.logspace(1, 4, 4, endpoint=True)
width = 1

for i, key in enumerate(keys):
    if i < len(labels)-3:
        ax = axes['a']
    else:
        ax = axes['b']
        ax.set_xlabel(xlabel_psd)
    if key:
        [f, s] = kid_dict[name][key]['noise']
        Q = kid_dict[name][key]['Q']
        ax.semilogx(f[exclude_dc:],  10*np.log10(s[exclude_dc:]/(4*Q**2)), label=labels[i], c=colors[i], ls=linestyles[i])
    else:
        ax.semilogx([],  [], label=labels[i], c=colors[i], ls=linestyles[i])
    ax.set_xlim(xlim_psd)
    ax.set_ylim(ylim_psd)
    ax.set_ylabel(ylabel_psd)
    ax.set_xticks(xticks, minor=False)
    ax.legend(loc='upper right', ncols=3, borderaxespad=0., frameon=False, handlelength=1, columnspacing=0.25)

plt.savefig('figures/%s_noise_spectra.pdf' % name)
plt.show()