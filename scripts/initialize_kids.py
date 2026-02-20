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
from .dipfit.KID_S21 import loop_over_S21_files
import matplotlib.pyplot as plt
import pickle
import os


def initialize_kid26(path2data, name, from_scratch=False):
    # -------------------------------------------------
    # Input all directories, KID numbers and readout powers for all measurements. 
    # -------------------------------------------------
    dir_bf_dark = path2data + r'LT218Chip1_BF_20230208_dark\12KIDs dark/'
    dir_38um_off = path2data + r'LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off/'
    dir_38um = path2data + r'LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800 long/'
    dir_85um = path2data + r'LT218Chip1_BF_20221025_MIR8_5\12KIDs mono off long/'
    dir_185um_off = path2data + r'LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB3K/'
    dir_185um = path2data + r'LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K/'
    dir_adr_dark = path2data + r'LT218Chip1_ADR_20240605_MIR25_dark\12KIDs_3Pread_100s_1MHz/'
    dir_25um_off = path2data + r'LT218Chip1_ADR_20240506_MIR25\12KIDs_3Pread_TD40s_1MHz_BB3K/'
    dir_25um = path2data + r'LT218Chip1_ADR_20240506_MIR25\12KIDs_3Pread_TD40s_1MHz_BB24K/'
    dir_mux = path2data + r'LT218_MUX_analysed\7kids_10000s_analysed\fastreadout_noise.00011_seg000.pkl'

                                  # general KID identification number. This is the detector analysed in the paper. Only the raw data for this detector is provided.
    wls = ['BF dark', '3.8um off', '3.8um', '8.5um', '18.5um off', '18.5um', 'ADR dark', '25um off', '25um', 'mux']
    dirs = [dir_bf_dark, dir_38um_off, dir_38um, dir_85um, dir_185um_off, dir_185um, dir_adr_dark, dir_25um_off, dir_25um, dir_mux]
    kids = [24, 24, 24, 24, 25, 25, 24, 24, 24, None]               
    preads = [113, 113, 113, 113, 117, 117, 113, 115, 115, None]    
    preads_s21 = [112, 113, 113, 113, 118, 118, 112, 116, 116, None] 
    models_s21 = 7*['khalilswensonbias'] + 2*['khalilswenson'] + [None]

    # -------------------------------------------------
    # Generate kid_dict
    # -------------------------------------------------
    file_path = '%skid_dict.pkl' % (path2data)
    if os.path.exists(file_path) and not from_scratch:
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
            kid_dict[name][wl]['model s21'] = models_s21[i]

    #--------------------------------------------------
    # Save kid_dict
    # -------------------------------------------------
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'wb') as f:
        pickle.dump(kid_dict, f)
    print("Updated file %s" % path2kid_dict)
