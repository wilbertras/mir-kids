from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_rawdata(filename, unit='wavenumber'):
    data = []
    name = filename
    data = pd.read_csv(name, header=None, dtype=float, sep=';|,', engine='python')
    data = np.array(data)
    data = data[np.argsort(data[:, 0])]
    if unit=='wavenumber':
        wavenum = data[:, 0]
        wl = 1e-2 / wavenum
    elif unit=='nm':
        wl = data[:, 0] * 1e-9
    elif unit=='um':
        wl = data[:, 0] * 1e-6
    theta = data[:, 1] * 0.01
    return wl, theta



def load_xls(xls, label, wls, in_out=None):
    data = np.array(pd.read_excel(xls, label, header=0, usecols=(0, 1)))
    x = 1/(data[:, 0]*1e2)
    y = data[:, 1]
    theta = interp1d(x, y, fill_value=in_out, bounds_error=False)(wls)
    return theta

def load_all_filters(wls=[]):
    if not len(wls):
        wls = np.logspace(-7, -3, 10000)

    wl_77k, theta_77k = load_rawdata('filters/38um/Sample piece 77k.csv')
    wl_nir, theta_nir = load_rawdata('filters/38um/3 - 08-20µm.CSV')
    wl_mir, theta_mir = load_rawdata('filters/38um/2 - 16-100um.CSV')
    wl_fir, theta_fir = load_rawdata('filters/38um/1 - 170-450µm.CSV')
    wl38 = np.hstack((wl_nir, wl_77k[wl_77k>np.nanmax(wl_nir)], wl_mir[wl_mir>np.nanmax(wl_77k)], wl_fir[wl_fir>np.nanmax(wl_mir)]))
    theta38 = np.hstack((theta_nir, theta_77k[wl_77k>np.nanmax(wl_nir)], theta_mir[wl_mir>np.nanmax(wl_77k)], theta_fir[wl_fir>np.nanmax(wl_mir)]))
    # wl, theta = load_rawdata('filters/38um/merged.CSV')
    # fig, ax = plt.subplots()
    # ax.plot(wl38, theta38)
    # ax.plot(wl, theta, ls='--')
    # ax.plot(wl_77k, theta_77k, ls='--')
    # plt.show()
    theta38 = interp1d(wl38, theta38, kind='linear', bounds_error=False, fill_value=(1e-2, 1e-2))(wls)
    theta38[theta38 < 1e-3] = 1e-3
    x_fir = np.array([70, 90, 100, 200, 300, 400, 500])*1e-6
    y_fir = np.array([0, .025, .05, .35, .6, .7, .8])
    irfs_fir = interp1d(x_fir, y_fir, kind='linear', bounds_error=False, fill_value=(0, .8))(wls)
    theta38 += irfs_fir

    wl_77k, theta_77k = load_rawdata('filters/85um/Sample piece 77k.csv')
    wl_nir, theta_nir = load_rawdata('filters/85um/3 - 0.8-2.0µm.CSV')
    wl_mir, theta_mir = load_rawdata('filters/85um/2 - 1.5-20um.CSV')
    wl_fir, theta_fir = load_rawdata('filters/85um/1 - 17.0-45.0µm.CSV')
    wl85 = np.hstack((wl_nir, wl_77k[wl_77k>np.nanmax(wl_nir)], wl_mir[wl_mir>np.nanmax(wl_77k)], wl_fir[wl_fir>np.nanmax(wl_mir)]))
    theta85 = np.hstack((theta_nir, theta_77k[wl_77k>np.nanmax(wl_nir)], theta_mir[wl_mir>np.nanmax(wl_77k)], theta_fir[wl_fir>np.nanmax(wl_mir)]))
    # wl, theta = load_rawdata('filters/85um/merged.CSV')
    # fig, ax = plt.subplots()
    # ax.plot(wl85, theta85)
    # ax.plot(wl, theta, ls='--')
    # ax.plot(wl_77k, theta_77k, ls='--')
    # plt.show()
    theta85 = interp1d(wl85, theta85, kind='linear', bounds_error=False, fill_value=(1e-2, .3))(wls)
    theta85[theta85 < 1e-3] = 1e-3

    wl_77k, theta_77k = load_rawdata('filters/185um/Sample piece 77k.csv', unit='nm')
    wl_mir, theta_mir = load_rawdata('filters/185um/1 - 1.5-17um.CSV', unit='nm')
    wl_fir, theta_fir = load_rawdata('filters/185um/2 - 16-40um.CSV', unit='nm')
    wl185 = np.hstack((wl_mir[wl_mir>np.nanmin(wl_77k)], wl_77k, wl_fir[wl_fir>np.nanmax(wl_77k)]))
    theta185 = np.hstack((theta_mir[wl_mir>np.nanmin(wl_77k)], theta_77k, theta_fir[wl_fir>np.nanmax(wl_77k)]))
    wl, theta = load_rawdata('filters/185um/merged.CSV')
    theta185 = interp1d(wl185, theta185, kind='linear', bounds_error=False, fill_value=(1e-3, .3))(wls)
    theta185[theta185 < 1e-3] = 1e-3
    # fig, ax = plt.subplots()
    # ax.plot(wls, theta185)
    # # ax.plot(wl, theta, ls='--')
    # # ax.plot(wl_77k, theta_77k, ls='--')
    # # ax.plot(wl_fir, theta_fir, ls='--')
    # ax.set_yscale('log')
    # plt.show()

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