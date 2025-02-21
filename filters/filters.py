from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.signal import savgol_filter
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

def load_all_filters():
    wls = np.logspace(-6, -3, 10000)

    wl38, theta38 = load_rawdata('filters/38um/merged.csv')
    theta38 = interp1d(wl38, theta38, kind='linear', bounds_error=False, fill_value=(1e-2, 1e-2))(wls)
    theta38[theta38 < 1e-3] = 1e-3
    theta38 = savgol_filter(theta38, 51, 3)
    x_fir = np.array([70, 90, 100, 200, 300, 400, 500])*1e-6
    y_fir = np.array([0, .025, .05, .35, .6, .7, .8])
    irfs_fir = interp1d(x_fir, y_fir, kind='linear', bounds_error=False, fill_value=(0, .8))(wls)
    theta38 += irfs_fir

    wl85, theta85 = load_rawdata('filters/85um/merged.csv')
    theta85 = interp1d(wl85, theta85, kind='linear', bounds_error=False, fill_value=(1e-2, .3))(wls)
    theta85[theta85 < 1e-3] = 1e-3

    wl185, theta185 = load_rawdata('filters/185um/merged.csv')
    theta185 = interp1d(wl185, theta185, kind='linear', bounds_error=False, fill_value=(1e-2, .3))(wls)
    theta185[theta185 < 1e-3] = 1e-3

    data = pd.read_csv('filters/CalciumFluoride/CaF2_Uncoated_Trans.csv', header=None, sep=';')
    wlcaf2 = data[0] / 1E6
    caf2 = data[1] / 1E2
    caf2 = interp1d(wlcaf2, caf2, kind='linear', bounds_error=False, fill_value=(1e-2, 1e-2))(wls)
    x_fir = np.array([100, 200, 300, 400, 500, 1000])*1e-6
    y_fir = np.array([0, .07, .14, .28, .35, .56])
    caf2_fir = interp1d(x_fir, y_fir, kind='linear', bounds_error=False, fill_value=(0, .8))(wls)
    caf2 += caf2_fir

    folder = 'filters/NeutralDensity/Neutral Density '
    data = pd.read_csv(folder + '1.csv', header=None, sep=';')
    wlnd1 = data[0] / 1E9
    nd1 = data[1] / 1E2
    nd1 = interp1d(wlnd1, nd1, kind='linear', bounds_error=False, fill_value=(1e-2, np.asarray(nd1)[-1]))(wls)

    data = pd.read_csv(folder + '2.csv', header=None, sep=';')
    wlnd2 = data[0] / 1E9
    nd2 = data[1] / 1E2
    nd2 = interp1d(wlnd2, nd2, kind='linear', bounds_error=False, fill_value=(1e-2, np.asarray(nd2)[-1]))(wls)
    nd2[np.isnan(nd2)] = 5e-4

    data = pd.read_csv(folder + '3.csv', header=None, sep=';')
    wlnd3 = data[0] / 1E9
    nd3 = data[1] / 1E2
    nd3 = interp1d(wlnd3, nd3, kind='linear', bounds_error=False, fill_value=(1e-3, np.asarray(nd3)[-1]))(wls)
    nd3[np.isnan(nd3)] = 3e-4

    folder = 'filters/Germanium/Uncoated_ge_window_rawData.csv'
    data = pd.read_csv(folder, header=None, sep=';')
    wlger = data[0] / 1E9
    ger = data[1] / 1E2
    ger = interp1d(wlger, ger, kind='linear', bounds_error=False, fill_value=(1e-2, .5))(wls)

    folder = 'filters/ZnSe/ZnSe.csv'
    data = pd.read_csv(folder, header=None, sep=';')
    wlznse = data[0] / 1E6
    wlznse = wlznse[300:]
    znse = data[1] / 1E2
    znse = znse[300:]
    znse = interp1d(wlznse, znse, kind='linear', bounds_error=False, fill_value=(1e-2, 1e-2))(wls)

    path = 'filters/25um/25um filterstack.xlsx'
    xls = pd.ExcelFile(path)
    filters = []
    labels = ['BP', 'HP300A', 'HP300B', 'LP600']
    in_outs = [(1e-3, .847), (.1, 1e-3), (.1, 1e-3), (1e-2, .95)]
    for i, label in enumerate(labels):
        filter = load_xls(xls, label, wls, in_outs[i])
        filter[filter<1e-4] = 1e-4
        filters.append(filter)
    [theta25, SP_A, SP_B, LP] = filters
    theta25[theta25 < 1e-4] = 1e-4

    [wlsi, si] = load_rawdata('filters/Si/plot-data.csv')
    si *= 1e2
    si = interp1d(wlsi, si, kind='linear', bounds_error=False, fill_value=(np.asarray(si)[0], np.asarray(si)[-1]))(wls)
    return {'wl':wls, 'bp38':theta38, 'bp85':theta85, 'bp185':theta185, 'caf2':caf2, 'nd1':nd1, 'nd2':nd2, 'nd3':nd3, 'ger':ger, 'znse':znse, 'bp25':theta25, 'sp_a':SP_A, 'sp_b':SP_B, 'lp':LP, 'si':si}

load_all_filters()