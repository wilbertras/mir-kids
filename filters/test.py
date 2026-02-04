from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.signal import savgol_filter, wiener
import numpy as np
import pandas as pd

def load_rawdata(filename):
    data = []
    name = filename
    data = pd.read_csv(name, header=None, dtype=float, sep=';|,', engine='python')
    data = np.array(data)
    # data = data[np.argsort(data[:, 0])]
    wavenum = data[:, 0]
    wl = 1e-2 / wavenum
    theta = data[:, 1] * 0.01
    return [wl, theta]


def load_xls(xls, label, wls, in_out=None):
    data = np.array(pd.read_excel(xls, label, header=0, usecols=(0, 1)))
    x = 1/(data[:, 0]*1e2)
    y = data[:, 1]
    theta = interp1d(x, y, fill_value=in_out, bounds_error=False)(wls)
    return theta

wls = np.logspace(-6, -3, 10000)

wl38, theta38 = load_rawdata('filters/38um/merged.csv')
theta38 = interp1d(wl38, theta38, kind='linear', bounds_error=False, fill_value=(1e-2, 1e-2))(wls)
# theta38[theta38 < 1e-3] = 1e-3
x_fir = np.array([70, 90, 100, 200, 300, 400, 500])*1e-6
y_fir = np.array([0, .025, .05, .35, .6, .7, .8])
irfs_fir = interp1d(x_fir, y_fir, kind='linear', bounds_error=False, fill_value=(0, .8))(wls)
theta38 += irfs_fir

fig, ax = plt.subplots()
ax.loglog(theta38)
ax.loglog(wiener(theta38, 31))
ax.loglog(savgol_filter(theta38, 31, 1, 0))
plt.show()