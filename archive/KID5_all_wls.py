import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlibcolors as matplotlibcolors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
plt.style.use('figures/matplotlibrc')

R_sn = [16,8,4]
wl = [3.8, 8.5, 18.5]
R = [5.7,4.9,1.8]
R_i = 1/np.sqrt(1/np.asarray(R)**2 - 1/np.asarray(R_sn)**2)
t_qp = [200, 219, 215]

fig, axes = plt.subplot_mosaic('ab', constrained_layout=True, sharex=True, figsize=(6, 3))
ax = axes['a']
ax.plot(wl, R, label='$R$', c='b', marker='o')
ax.plot(wl, R_sn, label='$R_{SN}$', c='o', marker='s')
ax.plot(wl, R_i, label='$R_{i}$', c='p', marker='^')
ax.set_xlabel('Wavelength [$\mu$m]')
ax.set_ylabel('Resolving power [-]')
ax.set_xlim([0,20])
ax.set_ylim([0,20])
ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
        ncols=2, mode="expand", borderaxespad=0., fontsize=9)
ax = axes['b']
ax.plot(wl, t_qp, label='$\\tau_{qp}$', c='p', marker='o')
ax.set_xlabel('Wavelength [$\mu$m]')
ax.set_ylabel('Lifetime [$\mu$s]')
ax.set_ylim([0,250])
ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
        ncols=2, mode="expand", borderaxespad=0., fontsize=9)
# plt.savefig('figures/KID5_all_wls.pdf')
plt.show()