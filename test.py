# %%
from main import pulse_analysis
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
# %%
    dir = "D:/Data/README Example Data/LT192Chip1_BF_20220235/1KIDs laser on 1545 50nW 46dB/TD_Power"
    kid = 1
    pread = 102
    file_type = 'vis'
    chuncksize = 10
    nr_chuncks = 1
    pw = 256
    pw_offset = 20
    filter = 'None'
    lifetime = 300
    mph = np.array([3, 15])
    mpp = mph[0]
    iterate = 0
    pulse_analysis(dir, kid, pread, file_type, chuncksize, nr_chuncks, pw, pw_offset, filter, lifetime, mph, mpp, iterate=iterate, plot=True, coord='circle', exclude_dc=True)

# %%
    dir = "D:/Data/README Example Data/LT218Chip1_BF_20240116_MIR18_5/12KIDS_185um_BB200K/TD_Power"
    kid = 5
    pread = 113
    file_type = 'vis'
    chuncksize = 10
    nr_chuncks = 4
    pw = 1500
    pw_offset = 100
    filter = 'exp'
    lifetime = 300
    mph = np.array([5, 20])
    mpp = 1
    iterate = 1
    pulse_analysis(dir, kid, pread, file_type, chuncksize, nr_chuncks, pw, pw_offset, filter, lifetime, mph, mpp, iterate=iterate, plot=True, coord='circle', exclude_dc=True)

# %%
