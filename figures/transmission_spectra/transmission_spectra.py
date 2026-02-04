import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from scipy.integrate import cumulative_trapezoid
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from filters.filters import load_all_filters
import matplotlibcolors

def planck_wl(wl, t):
    return sc.h * sc.c**2 / (wl**5 * (np.exp(sc.h * sc.c / (wl * sc.k * t)) - 1))

filters = load_all_filters()

wl = filters['wl']*1e6
bp38 = filters['bp38']
bp85 = filters['bp85']
bp185 = filters['bp185']
caf2 = filters['caf2']
znse = filters['znse']
nd1 = filters['nd1']
nd2 = filters['nd2']
nd3 = filters['nd3']
ger = filters['ger'] 
bp25 = filters['bp25']
sp_a = filters['sp_a']
sp_b = filters['sp_b']
lp = filters['lp']
si = filters['si']

A = (1e-3)**2
omega = np.pi*(10e-3)**2/(300e-3)**2
tot38 = bp38**4*caf2**4*ger*nd2*nd3*si*A*omega
tot85 = bp85**4*caf2**4*ger*nd2*nd3*si*A*omega
tot185 = bp185**5*nd1*nd3*znse**2*si*A*omega
tot25 = bp25**3*lp*sp_a**3*sp_b*si*A*omega

bb300 = planck_wl(wl*1e-6, 300)*1e-6
bb160 = planck_wl(wl*1e-6, 160)*1e-6
bb40 = planck_wl(wl*1e-6, 40)*1e-6

fig, axes = plt.subplot_mosaic('ab;cd;ef', figsize=(12,5), constrained_layout=True, sharex=True)
xlim = [1, 100]
ax = axes['a']
ax.semilogx(wl, bp38, label='BP38')
ax.fill_between(wl, bp38, 0, where=(wl < 45), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

ax = axes['b']
ax.semilogx(wl, bp85, label='BP85', zorder=1)
ax.fill_between(wl, bp85, 0, where=(wl < 45), alpha=0.3)
ax.semilogx(wl, filters['caf2'], label='$CaF_2$', zorder=0)
ax.fill_between(wl, caf2, 0, where=(wl < 16), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('Transmission [-]')
# ax.set_yticklabels([])

ax = axes['c']
ax.semilogx(wl, bp185, label='BP185', zorder=1)
ax.fill_between(wl, bp185, 0, where=(wl < 40), alpha=0.3)
ax.semilogx(wl, znse, label='ZnSe', zorder=0)
ax.fill_between(wl, znse, 0, where=(wl < 25), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

ax = axes['d']
ax.semilogx(wl, bp25, label='BP25', zorder=3)
ax.fill_between(wl, bp25, 0, where=((wl < 77) & (wl > 1.67)), alpha=0.3)
ax.semilogx(wl, sp_a, label='SP1', zorder=2)
ax.fill_between(wl, sp_a, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
ax.semilogx(wl, sp_b, label='SP2', zorder=1)
ax.fill_between(wl, sp_b, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
ax.semilogx(wl, lp, label='LP', zorder=0)
ax.fill_between(wl, lp, 0, where=((wl < 82)&(wl > 1.67)), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('Transmission [-]')
# ax.set_yticklabels([])

ax = axes['e']
ax.plot(wl, si, label='Si')
ax.fill_between(wl, si, 0, alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')
# ax.set_yticklabels([])

ax = axes['f']
ax.loglog(wl, nd1, label='ND1')
ax.fill_between(wl, nd1, 0, where=(wl < 18), alpha=0.3)
ax.loglog(wl, nd2, label='ND2')
ax.fill_between(wl, nd2, 0, where=(wl < 18), alpha=0.3)
ax.loglog(wl, nd3, label='ND3')
ax.fill_between(wl, nd3, 0, where=(wl < 18), alpha=0.3)
ax.legend()
ax.set_ylim([1e-4, 1e0])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('Transmission [-]')


fig.savefig('filter_transmissions.pdf')

fig, axes = plt.subplot_mosaic('ab;cd', figsize=(12,5), constrained_layout=True, sharey=True, sharex=True)
ax = axes['a']
ax.loglog(wl, tot38*bb300, label='3.8 µm')
ax.fill_between(wl, tot38*bb300, 0, alpha=0.3)
ax.loglog(wl, bb300* A * omega, c='k')
ax.annotate('Planck 300 K', xy=(10,0.01), xycoords='data', size=8)
cumulative_tot38_bb300 = np.cumsum(tot38 * bb300*np.diff(wl, prepend=wl[0]))
# cumulative_tot38_bb300 = cumulative_trapezoid(tot38 * bb300, wl, initial=None)
ax.loglog(wl, cumulative_tot38_bb300, label='Cumulative power')
ax.legend()
ax.set_ylim([1e-30, 1e-5])
ax.set_xlim([1,100])
# ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')

ax = axes['b']
ax.loglog(wl, tot85*bb300, label='8.5 µm')
ax.fill_between(wl, tot85*bb300, 0, alpha=0.3)
ax.loglog(wl, bb300* A * omega, c='k')
ax.annotate('Planck 300 K', xy=(10,0.01), xycoords='data', size=8)
cumulative_tot85_bb300 = cumulative_trapezoid(tot85 * bb300, wl, initial=None)
ax.loglog(wl[1:], cumulative_tot85_bb300, label='Cumulative power')
ax.legend()
# ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')

ax = axes['c']
ax.loglog(wl, tot185*bb160, label='18.5 µm')
ax.fill_between(wl, tot185*bb160, 0, alpha=0.3)
ax.loglog(wl, bb160* A * omega, c='k')
ax.annotate('Planck 160 K', xy=(18.5,0.001), xycoords='data', size=8)
cumulative_tot185_bb160 = cumulative_trapezoid(tot185 * bb160, wl, initial=None)
ax.loglog(wl[1:], cumulative_tot185_bb160, label='Cumulative power')
ax.legend()
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')

ax = axes['d']
ax.loglog(wl, tot25*bb40, label='25 µm')
ax.fill_between(wl,  tot25*bb40, 0, alpha=0.3)
ax.loglog(wl, bb40* A * omega, c='k')
ax.annotate('Planck 40 K', xy=(25,0.001), xycoords='data', size=8)
cumulative_tot25_bb40 = cumulative_trapezoid(tot25 * bb40, wl, initial=None)
ax.loglog(wl[1:], cumulative_tot25_bb40, label='Cumulative power')
ax.legend()
ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')
fig.savefig('transmission_spectra.pdf')
plt.show()