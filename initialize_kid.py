"""
initialize_kid.py

This module initializes a dictionary 'kid_dict' that stores all the analysed measurement data. 
The setup of the code is general but is specifically tailored to analyse KID26, which is the detector analysed in the paper.

In this module the following data is added to kid_dict:
    - 'name': Global KID identification number. Keys of this dictionary are all the differerent measurement configurations:
             'BF dark', '3.8um off', '3.8um', '8.5um', '18.5um off', '18.5um', 'ADR dark', '25um off', '25um', 'mux' 
             for any configuration 'x', the following data is added:
        - 'x': any configuration
            - 'kid': KID number specific to measurement configuration. The KID number for this specific measurement configuration might differ from the global KID number
            - 'dir': directory of the measurement data for this configuration
            - 'Q': fitted quality factor for this configuration, important for scaling the responses appropriately
            - 'pread': readout power for time domain pulse data
            - 'pread s21': readout power used for fitting the KID dip. This might differ marginally with +-1 dBm from 'pread' as the KID dip fitting is more sensitive to the readout power than the pulse counting. The readout power used for fitting the KID dips is chosen based on the best fit of the KID dip.
"""

#--------------------------------------------------
# Import modules
# -------------------------------------------------
from S21.KID_S21 import loop_over_S21_files
import matplotlib.pyplot as plt
import pickle
import os

# -------------------------------------------------
# Input all directories, KID numbers and readout powers for all measurements. 
# -------------------------------------------------
with open('path2data.txt', 'r') as file:
    main_dir = file.readlines()[0]
dir_bf_dark = main_dir + r'LT218Chip1_BF_20230208_dark\12KIDs dark/'
dir_38um_off = main_dir + r'LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off/'
dir_38um = main_dir + r'LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800 long/'
dir_85um = main_dir + r'LT218Chip1_BF_20221025_MIR8_5\12KIDs mono off long/'
dir_185um_off = main_dir + r'LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB3K/'
dir_185um = main_dir + r'LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K/'
dir_adr_dark = main_dir + r'LT218Chip1_ADR_20240605_MIR25_dark\12KIDs_3Pread_100s_1MHz/'
dir_25um_off = main_dir + r'LT218Chip1_ADR_20240506_MIR25\12KIDs_3Pread_TD40s_1MHz_BB3K/'
dir_25um = main_dir + r'LT218Chip1_ADR_20240506_MIR25\12KIDs_3Pread_TD40s_1MHz_BB24K/'
dir_mux = main_dir + r'LT218_MUX_analysed\7kids_10000s_analysed\fastreadout_noise.00011_seg000.pkl'

name = 'KID26'                                      # general KID identification number. This is the detector analysed in the paper. Only the raw data for this detector is provided.
wls = ['BF dark', '3.8um off', '3.8um', '8.5um', '18.5um off', '18.5um', 'ADR dark', '25um off', '25um', 'mux']
dirs = [dir_bf_dark, dir_38um_off, dir_38um, dir_85um, dir_185um_off, dir_185um, dir_adr_dark, dir_25um_off, dir_25um, dir_mux]
kids = [24, 24, 24, 24, 25, 25, 24, 24, 24, None]               
preads = [113, 113, 113, 113, 117, 117, 113, 115, 115, None]    
preads_s21 = [112, 113, 113, 113, 118, 118, 112, 116, 116, None] 

# -------------------------------------------------
# Generate kid_dict
# -------------------------------------------------
file_path = '%skid_dict.pkl' % (main_dir)
if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        kid_dict = pickle.load(f)
    print("Loaded file %s" % file_path)
else:
    print(f"{file_path} does not exist. Initiated empty dictionary.")
    kid_dict = {}
    kid_dict[name] = {} 
    for i, wl in enumerate(wls):
        kid_dict[name][wl] = {}
        kid_dict[name][wl]['dir'] = dirs[i]
        kid_dict[name][wl]['kid'] = kids[i]
        kid_dict[name][wl]['pread'] = preads[i]
        kid_dict[name][wl]['pread s21'] = preads_s21[i]

# -------------------------------------------------
# Fit loaded quality factors
# -------------------------------------------------
models = 7*['khalilswensonbias'] + 2*['khalilswenson']
for i, wl in enumerate(wls[:-1]):
    df = loop_over_S21_files(dirs[i] + 'FFT/Power/', kids[i], pread=preads_s21[i], plot=True, model=models[i], method='least squares', append='__S21dB')
    kid_dict[name][wl]['Q'] = df['Ql'].iloc[0]
    print('\n', wl)
    print('\n', df, '\n')

wl ='mux'
with open(dir_mux, 'rb') as f:
    mux = pickle.load(f)
kid_dict[name][wl]['Q'] = mux[name]['Q']
kid_dict[name][wl]['nxx'] = mux[name]['Nxx']

# -------------------------------------------------
# Save kid_dict
# -------------------------------------------------
with open(file_path, 'wb') as f:
    pickle.dump(kid_dict, f)
    print("Saved file %s" % file_path)


plt.show()