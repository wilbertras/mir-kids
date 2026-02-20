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


def initialize_quality_factors(path2data, name):
    #--------------------------------------------------
    # Load kid_dict
    # -------------------------------------------------
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'rb') as f:
        kid_dict = pickle.load(f)
    print("Loaded file %s" % path2kid_dict)

    # -------------------------------------------------
    # Fit loaded quality factors
    # -------------------------------------------------
    for key, item in kid_dict[name].items():
        if key == 'mux':
            dir_mux = item['dir']
        else:
            dir = item['dir']
            kid = item['kid']
            pread_s21 = item['pread s21']
            model = item['model s21']
            df = loop_over_S21_files(dir + 'FFT/Power/', kid, pread=pread_s21, plot=True, model=model, method='least squares', append='__S21dB')
            item['Q'] = df['Ql'].iloc[0]
            print('\n', key, ':\n', df)

    #--------------------------------------------------
    # Save kid_dict
    # -------------------------------------------------
    with open(path2kid_dict, 'wb') as f:
        pickle.dump(kid_dict, f)
    print("Updated file %s" % path2kid_dict)

    plt.show()


# Code that should only run when this file is executed directly
if __name__ == "__main__":
    print("This code runs only when initialize_kid.py is executed directly.")
    initialize_quality_factors()