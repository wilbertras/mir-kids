import numpy as np
import natsort
import glob
import pandas as pd
import re
import io
import matplotlib.pyplot as plt

from ..dipfit.Khalil import KhalilModel_magspace, KhalilSwensonModel, KhalilSwensonModelBias


def def_Pint(Q,Qe,Pread):
        Pint = 10*np.log10((1/np.pi)*(Q**2/Qe)) + Pread
        return Pint

def create_result_pd():
    df_results = pd.DataFrame(columns=["KID", "Power", "Temperature",
                                       "f0", 'f0_std',
                                       "Ql", "Ql_std",
                                       "Qc", "Qc_std",
                                       "Qi", "Qi_std",
                                       "Pint"])
    return df_results
    
def add_result(df_results, kid, power, temperature, fit_result, Pint, phi):
    new_entry = pd.DataFrame({"KID": [kid], "Power": [power], "Temperature": [temperature],
                              "f0": [fit_result.params['f0'].value],
                              "f0_std": [fit_result.params['f0'].stderr],
                              "Ql": [fit_result.params['Ql'].value],
                              "Ql_std": [fit_result.params['Ql'].stderr],
                              "Qc": [fit_result.params['Qc_re'].value],
                              "Qc_std": [fit_result.params['Qc_re'].stderr],
                              "Qi": [fit_result.params['Qi'].value],
                              "Qi_std": [fit_result.params['Qi'].stderr],
                              "Pint": [Pint],
                              "redchisqr": [fit_result.redchi],
                              "phi": [phi]})
    
    for param in ['a_nonlin', 'dw']:
        if param in fit_result.params:
            new_entry[param] = [fit_result.params[param].value]
            new_entry[param + '_std'] = [fit_result.params[param].stderr]
        # else:
        #     new_entry[param] = [np.nan]
        #     new_entry[param + '_std'] = [np.nan]
    if df_results.empty or df_results.isna().all().all():
        df_results = new_entry
    else:
        df_results = pd.concat([df_results, new_entry], ignore_index=True)
    return df_results

def find_S21_files(path, kid='', pread='', append=''):

    string = path + 'KID%s_%sdBm*%s.dat' % (kid,pread,append)    
    files = natsort.natsorted(glob.glob(string)) # a sorted lis of filenames in path
    nr_files = len(files)
    
    kid = []
    for file in files:
        kid.append(int(re.findall(r"KID(\d+)_", file)[0]))
        # kid.append(int(re.findall(path + "KID(\d+)_(\d{2,3})dBm_", file)[0]))
        
    kids = np.unique(kid)
    
    return files, kids
    # optional print some info like path name and nr of KIDs
    
def loop_over_S21_files(path, kid=None, pread=None, model=None, plot=False, append='', method=None, guess_Q=None):
    if not kid:
        kid = '*'
    else:
        kid = str(kid)
    if not pread:
        pread = '*'
    else:
        pread = str(pread)

    filenames, kids = find_S21_files(path, kid, pread, append)
    # output_dir = "temperature_data"
    # os.makedirs(output_dir, exist_ok=True)
    
    df_results = create_result_pd()

    for i, file_path in enumerate(filenames):
        [kid, power] = re.findall("KID(\d+)_(\d+)dBm_", file_path)[0]
        kid_id = int(kid)
        Pread = -float(power)
        
        with open(file_path, 'r') as file:
            file_contents = file.readlines()

        # Extract data organized by temperature
        data_by_temperature = preprocess_file_fast(file_contents)
                
        all_temperatures = list(data_by_temperature.keys())
        plot_temperatures = all_temperatures[::5]
        
        for temperature, df in data_by_temperature.items():
            #print(f"Temperature: {temperature} K")

            # Extract Frequency and S21 data
            frequencies = df['Frequency'].values
            s21_values = df['dB'].values
            rad_values = df['Rad'].values
        
            if (np.min(frequencies) != np.max(frequencies)):
        
                # here we start fitting the data
                result = Fit_S21(frequencies, s21_values, model, method, guess_Q)

                S21_fit_line = result.eval()
                if plot:
                    if i % plot == 0:
                        fig, ax = plt.subplots(figsize=(4,3))
                        result.plot_fit(ax)
                Pint = def_Pint(result.params['Ql'].value, result.params['Qc_re'].value, Pread)
                phi = np.arctan(2*result.params['Ql'].value*result.params['dw']/result.params['f0'].value)
                
                df_results = add_result(df_results, kid_id, Pread, temperature, result, Pint, phi)
                
                df['Mag'] = 10**((s21_values - np.mean(s21_values[0:100]))/20)
                df['Fit'] = S21_fit_line        
    return df_results

            
def Fit_S21(f, S21_dB, model, method=None, dw_low_power=None, guess_Q=None):
    S21_dB = S21_dB - np.mean(S21_dB[0:100]) # normalize the data
    
    S21_mag = 10**(S21_dB/20)
    # load the models --------|KHALILSWENSONMODEL TOEGEVOEGD|------------
    if model is not None and model.lower().strip().replace(" ", "").replace("_", "") in ("khalilswenson", "khalilswensonmodel", "swensonkhalil", "swensonkhalilmodel"): 
        Model_mag = KhalilSwensonModel
    elif model is not None and model.lower().strip().replace(" ", "").replace("_", "") in ("khalilswensonbias", "khalilswensonmodelbias", "swensonkhalilbias", "swensonkhalilmodelbias"): 
        Model_mag = KhalilSwensonModelBias
    else:
        Model_mag = KhalilModel_magspace
    model_mag = Model_mag(f, S21_mag, guess_Q)

# -------------------------------|dw-WAARDE FORCEREN|-------------
    if dw_low_power is not None:
        params = model_mag.guess
        params["dw"].set(value=dw_low_power, vary=False)
    
    
    # a first estimate of the fit -------------| USE PASSED METHOD (e.g. 'least_squares') AND PROPAGATE NANS |-------------------------------------
    if method: 
        result_pre = model_mag.fit(S21_mag, f=f, params = model_mag.guess, method=method, nan_policy='propagate')
    else: 
        result_pre = model_mag.fit(S21_mag, f=f, params = model_mag.guess, nan_policy='propagate')        
    return result_pre
    

def preprocess_file_fast(file_contents):
    data_by_temperature = {}
    current_temperature = None
    current_data = []

    # Process each line, reducing Python overhead
    for line in file_contents:
        # If temperature is found
        temp_match = re.match(r'Temperature in K:(\d+\.\d+)', line)
        if temp_match:
            if current_temperature is not None and current_data:
                # Convert current_data to a pandas DataFrame
                data_str = "\n".join(current_data)
                df = pd.read_csv(io.StringIO(data_str), sep='\t', header=None, names=['Frequency', 'dB', 'Rad'])
                data_by_temperature[current_temperature] = df
            
            # Set new temperature and reset current data
            current_temperature = float(temp_match.group(1))
            current_data = []
        
        # Collect tab-separated data (GHz, dB, Rad)
        elif re.match(r'\d+\.\d+E?[+-]?\d+?\t', line):
        
            current_data.append(line.strip())
    
    # Save the last block of data
    if current_temperature is not None and current_data:
        data_str = "\n".join(current_data)
        df = pd.read_csv(io.StringIO(data_str), sep='\t', header=None, names=['Frequency', 'dB', 'Rad'])
        data_by_temperature[current_temperature] = df

    return data_by_temperature