'''
Python implementation of the Matlab script that fits |S21| data to the Khalil model.
I checked that it gives the same results as the Matlab script. 

I experiment with a new version in the V2 functions.
'''
import lmfit as lmf
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate





def estimate_initials(f, mag):
    a_guess = np.average([mag[-1], mag[0]]) 
    b_guess = (mag[-1] - mag[0]) / (f[-1] - f[0])
    mag_c = mag / a_guess
    min_index = np.argmin(mag_c)
    f0_guess = f[min_index]
    S21min_guess = mag_c[min_index]
    bwindex = np.argmin(np.abs(mag_c**2-((S21min_guess**2+1)/2)))
    if bwindex-min_index==0:
        Ql_guess=f0_guess/(np.abs(2*(f[min_index+1]-f[min_index])))
    else:
        Ql_guess=f0_guess/(np.abs(2*(f[bwindex]-f[min_index])))
        
    Qi_guess = Ql_guess/S21min_guess
    Qc_guess = (Ql_guess*Qi_guess)/(Qi_guess - Ql_guess)
    BW_guess = f0_guess/Ql_guess
    return Ql_guess, f0_guess, Qc_guess, a_guess, b_guess, BW_guess

def Khalil_func_magspace(f, f0, Ql, Qc_re, dw, a, b):
    return np.abs(1 - (Ql/Qc_re *(1 + 2j*Ql*dw/f0) / (1 + 2j * Ql * (f - f0) / f0)))*np.abs((b*(f-f0)+a))

def Khalil_func_magspace_noslope(f, f0, Ql, Qc_re, dw, a, b):
    return np.abs(1 - (Ql/Qc_re *(1 + 2j*Ql*dw/f0) / (1 + 2j * Ql * (f - f0) / f0)))

def Khalil_func_logspace(f, f0, Ql, Qc_re, dw, a, b):
    return 20*np.log10(np.abs(1 - (Ql/Qc_re *(1 + 2j*Ql*dw/f0) / (1 + 2j * Ql * (f - f0) / f0)))*np.abs((b*(f-f0)+a)))


# ----------------| KHALILSWENSON FITFUNCTIE |---------------
def khalil_swenson(f, f0, Ql, Qc_re, a_nonlin, dw):
    wgs = f
    w0 = f0
    Q = Ql
    # Q = 1/(1/Qi + 1/Qc)
    Qc = Qc_re
    a = a_nonlin  

    x0 = (wgs - w0)/w0
    y0 = Q * x0
    discriminant = -( (4*y0**2 - 3)**3 )/46656 + ( (27*a + 8*y0**3 + 18*y0)**2 )/46656 # Cardano

    # solution for discriminant > 0, np.sqrt may return nan for fits of data beyond bifurcation point. 
    # That's ok, don't replace np.sqrt by complex square root.
    #  np.cbrt only returns any real roots, that's ok for discriminants > 0, i.e. for fits outside of the bifurcation region.
    t = (  np.cbrt(-np.sqrt(discriminant) + a/8 + y0**3/27 + y0/12)
            + np.cbrt( np.sqrt(discriminant) + a/8 + y0**3/27 + y0/12))

    y = t + y0/3
    x = y/Q                                                         # Swenson result
    S21 = np.abs(1 - (Q/Qc * (1 + 2j * Q * dw/w0))/(1 + 2j*Q*x))    # Substitute into Khalil model.
    S21 = np.where(np.isfinite(S21), S21, 1e12)                     # replace infs with 1e12 to keep 'least_squares' fitting method happy.
    return S21

# ----------------| KHALILSWENSON FITFUNCTIE BIAS |---------------
def khalil_swenson_bias(f, f0, Ql, Qc_re, a_nonlin, dw, a, b):
    wgs = f
    w0 = f0
    Q = Ql
    # Q = 1/(1/Qi + 1/Qc)
    Qc = Qc_re

    x0 = (wgs - w0)/w0
    y0 = Q * x0
    discriminant = -( (4*y0**2 - 3)**3 )/46656 + ( (27*a_nonlin + 8*y0**3 + 18*y0)**2 )/46656 # Cardano

    # solution for discriminant > 0, np.sqrt may return nan for fits of data beyond bifurcation point. 
    # That's ok, don't replace np.sqrt by complex square root.
    #  np.cbrt only returns any real roots, that's ok for discriminants > 0, i.e. for fits outside of the bifurcation region.
    t = (  np.cbrt(-np.sqrt(discriminant) + a_nonlin/8 + y0**3/27 + y0/12)
            + np.cbrt( np.sqrt(discriminant) + a_nonlin/8 + y0**3/27 + y0/12))

    y = t + y0/3
    x = y/Q                                                         # Swenson result
    S21 = np.abs(1 - (Q/Qc * (1 + 2j * Q * dw/w0))/(1 + 2j*Q*x)) * np.abs((b*(wgs-w0)+a))    # Substitute into Khalil model.
    S21 = np.where(np.isfinite(S21), S21, 1e12)                     # replace infs with 1e12 to keep 'least_squares' fitting method happy.
    return S21

# --------------| KHALILSWENSON MODEL CLASS |---------------------------
class KhalilSwensonModel(lmf.model.Model):
    def __init__(self, f, mag, *args, **kwargs):
        super().__init__(khalil_swenson, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, _, _, BW_guess = estimate_initials(f, mag)
       
        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_re', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('dw')
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('a_nonlin', min = 0, max = 10)
        self.set_param_hint('Qi', expr='1/(1/Ql - 1/Qc_re)') # For convenience, does not affect the fit!
        
        params = self.make_params(Ql=Ql_guess, Qc_re=Qc_norm_guess, dw=0, f0=f0_guess, 
                                    a_nonlin_guess=0)
    
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)    

# --------------| KHALILSWENSONBIAS MODEL CLASS |---------------------------
class KhalilSwensonModelBias(lmf.model.Model):
    def __init__(self, f, mag, guess_Q=None, *args, **kwargs):
        super().__init__(khalil_swenson_bias, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, b_guess, BW_guess = estimate_initials(f, mag)
        if guess_Q is not None:
            Ql_guess = Qc_norm_guess = guess_Q

        self.set_param_hint('Ql', min = .1*Ql_guess, max=20*Ql_guess)
        self.set_param_hint('Qc_re', min = .1*Qc_norm_guess, max=20*Qc_norm_guess)
        self.set_param_hint('dw')
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('a_nonlin', min = 0, max = 10)
        self.set_param_hint('a', min = 0, max = 100)
        self.set_param_hint('b', min = -1e6, max = 1e6)
        self.set_param_hint('Qi', expr='1/(1/Ql - 1/Qc_re)') # For convenience, does not affect the fit!
        params = self.make_params(Ql=Ql_guess, Qc_re=Qc_norm_guess, dw=0, f0=f0_guess, 
                                    a_nonlin_guess=0, a=a_guess, b=b_guess)
    
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)    

class KhalilModel_magspace(lmf.model.Model):
    # magnitude version of the Khalil fit
    def __init__(self, f, mag, *args, **kwargs):
        super().__init__(Khalil_func_magspace, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, b_guess, BW_guess = estimate_initials(f, mag)

        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_re', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('dw')
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('a', min = 0, max = 100)
        self.set_param_hint('b', min = -1e6, max = 1e6)
        self.set_param_hint('Qi', expr='1/(1/Ql - 1/Qc_re)') # For convenience, does not affect the fit!
        
        params = self.make_params(Ql=Ql_guess, Qc_re=Qc_norm_guess, dw=0, f0=f0_guess, 
                                  a=a_guess, b=b_guess)
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)    

class KhalilModel_magspace_noslope(lmf.model.Model):
    # magnitude version of the Khalil fit
    def __init__(self, f, mag, *args, **kwargs):
        super().__init__(Khalil_func_magspace_noslope, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, b_guess, BW_guess = estimate_initials(f, mag)
       
        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_re', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('dw')
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('Qi', expr='1/(1/Ql - 1/Qc_re)') # For convenience, does not affect the fit!
        
        params = self.make_params(Ql=Ql_guess, Qc_re=Qc_norm_guess, dw=0, f0=f0_guess)
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)   

class KhalilModel_logspace(lmf.model.Model):
    # magnitude version of the Khalil fit
    def __init__(self, f, mag, *args, **kwargs):
        super().__init__(Khalil_func_logspace, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, b_guess, BW_guess = estimate_initials(f, 10**(mag/20))
       
        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_re', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('dw')
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('a', min = 0, max = 100)
        self.set_param_hint('b', min = -1e6, max = 1e6) 
        self.set_param_hint('Qi', expr='1/(1/Ql - 1/Qc_re)') # For convenience, does not affect the fit!
        
        params = self.make_params(Ql=Ql_guess, Qc_re=Qc_norm_guess, dw=0, f0=f0_guess, 
                                  a=a_guess, b=b_guess)
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)

## some new functions that will fit the skew

def y_nonlin(y0,a):
    
#     y1 = 1/48*(16*y0+((-4-4*1j*np.sqrt(3))*(-3+4*y0**2))/(27*a+18*y0+8*y0**3+3*np.sqrt(3)*np.sqrt(27*a**2+(1+4*y0**2)**2+4*a*y0*(9+4*y0**2)))**(1/3)+4*1j*(1j+np.sqrt(3))*(27*a+18*y0+8*y0**3+3*np.sqrt(3)*np.sqrt(27*a**2+(1+4*y0**2)**2+4*a*y0*(9+4*y0**2)))**(1/3))
        
#     if np.isnan(y1).any():
#         ind_nan = np.argwhere(np.isnan(y1))
#         print('nan in y1, y0: ' + str(y0[ind_nan]) + ', a: ' + str(a))
        
#     y2 = 1/24*(8*y0+(4*(-3+4*y0**2))/(27*a+18*y0+8*y0**3+3*np.sqrt(3)*np.sqrt(27*a**2+(1+4*y0**2)**2+4*a*y0*(9+4*y0**2)))**(1/3)+4*(27*a+18*y0+8*y0**3+3*np.sqrt(3)*np.sqrt(27*a**2+(1+4*y0**2)**2+4*a*y0*(9+4*y0**2)))**(1/3))
        
#     if np.isnan(y2).any():
#         ind_nan = np.argwhere(np.isnan(y2))
#         print('nan in y2, y0: ' + str(y0[ind_nan]) + ', a: ' + str(a))
    # test_for_zero = [i for i in range(len(y0)) if y0[i] == 0]
    # if len(test_for_zero) > 0:
    #     print('found a zero in y0')

    y1 = 1/48*(16*y0+((-4-4*1j*np.sqrt(3))*(-3+4*y0**2))/(27*a+18*y0+8*y0**3+3*np.sqrt(3)*np.sqrt(27*a**2+(1+4*y0**2)**2+4*a*y0*(9+4*y0**2)))**(1/3)+4*1j*(1j+np.sqrt(3))*(27*a+18*y0+8*y0**3+3*np.sqrt(3)*np.sqrt(27*a**2+(1+4*y0**2)**2+4*a*y0*(9+4*y0**2)))**(1/3))

    y2 = 1/24*(8*y0+(4*(-3+4*y0**2))/(27*a+18*y0+8*y0**3+3*np.sqrt(3)*np.sqrt(27*a**2+(1+4*y0**2)**2+4*a*y0*(9+4*y0**2)))**(1/3)+4*(27*a+18*y0+8*y0**3+3*np.sqrt(3)*np.sqrt(27*a**2+(1+4*y0**2)**2+4*a*y0*(9+4*y0**2)))**(1/3))

    y = y1.copy()
    y[np.imag(y1)>1e-1] = y2[np.imag(y1)>1e-1]
    
    if a < 4*np.sqrt(3)/9:
        try:
            last_index_y1_valid = [i for i in range(len(y1)) if np.imag(y1[i]) > 1e-1][0]-1
            range_patch = 15
            range_interp = 5
            y0_interp = np.concatenate([y0[last_index_y1_valid-(range_patch+range_interp):last_index_y1_valid-(range_patch-range_interp-1)], y0[last_index_y1_valid+(range_patch-range_interp-1):last_index_y1_valid+(range_patch+range_interp)]])
            y1_interp = np.concatenate([y[last_index_y1_valid-(range_patch+range_interp):last_index_y1_valid-(range_patch-range_interp-1)], y[last_index_y1_valid+(range_patch-range_interp-1):last_index_y1_valid+(range_patch+range_interp)]])
            f = interpolate.interp1d(y0_interp, y1_interp, fill_value='extrapolate')
            y[last_index_y1_valid-range_patch:last_index_y1_valid+range_patch] = f(y0[last_index_y1_valid-range_patch:last_index_y1_valid+range_patch])
        except:
            flag = 1
        
        
    if np.isnan(y).any():
        ind_nan = np.argwhere(np.isnan(y))
        y[ind_nan] = y[ind_nan+1]
        if np.isnan(y).any():
            print('nan in y after patch, y0: ' + str(y0[ind_nan]) + ', a: ' + str(a))
            fig, ax = plt.subplots()
            ax.plot(y0, y, '-')
            ax.plot(y0[ind_nan],y1[ind_nan], 'o')
            ax.plot(y0[ind_nan-1],y1[ind_nan-1], 'o')
            ax.plot(y0[ind_nan+1],y1[ind_nan+1], 'o')
            fig.show
        
    if np.isnan(y).any():
        ind_nan = np.argwhere(np.isnan(y))
        print('nan in y, y0: ' + str(y0[ind_nan]) + ', a: ' + str(a))
    
    return y

def skew_func_magspace(f, f0, Ql, Qc_re, dw, a, b, a_nonlin):
    
    x0 = (f - f0)/f0
    y0 = Ql*x0
    
    y = np.real(y_nonlin(y0.astype(complex), a_nonlin))
    
    x = np.nan_to_num(y/Ql)
    
    return np.abs(1 - (Ql/Qc_re *(1 + 2j*Ql*dw/f0) / (1 + 2j * Ql * x))) *np.abs((b*(f-f0)+a)) 
        
        
def skew_func_logspace(f, f0, Ql, Qc_re, dw, a, b, a_nonlin):
    
    x0 = (f - f0)/f0
    y0 = Ql*x0
    
    y = np.real(y_nonlin(y0.astype(complex), a_nonlin))
    
    x = np.nan_to_num(y/Ql)
    
    return 20*np.log10(np.abs(1 - (Ql/Qc_re *(1 + 2j*Ql*dw/f0) / (1 + 2j * Ql * x))*np.abs((b*(f-f0)+a)))*np.abs((b*(f-f0)+a)))

class Skewmodel_magspace(lmf.model.Model):
    # magnitude version of the Khalil fit
    def __init__(self, f, mag, *args, **kwargs):
        super().__init__(skew_func_magspace, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, b_guess, BW_guess = estimate_initials(f, mag)
       
        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_re', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('dw')
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('Qi', expr='1/(1/Ql - 1/Qc_re)') # For convenience, does not affect the fit!
        self.set_param_hint('a', min = 0, max = 100)
        self.set_param_hint('b', min = -1e6, max = 1e6)
        self.set_param_hint('a_nonlin', min = 1e-18, value = 1e-8)
 
        params = self.make_params(Ql=Ql_guess, Qc_re=Qc_norm_guess, dw=0, f0=f0_guess, a=a_guess, b=b_guess, a_nonlin = 0.01)
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)                        

class Skewmodel_logspace(lmf.model.Model):
    # attampt to fit the skew
    def __init__(self, f, mag, *args, **kwargs):
        super().__init__(skew_func_logspace, *args, **kwargs)

        Ql_guess, f0_guess, Qc_norm_guess, a_guess, b_guess, BW_guess = estimate_initials(f, 10**(mag/20))
       
        self.set_param_hint('Ql', min = 0.1*Ql_guess, max=10*Ql_guess)
        self.set_param_hint('Qc_re', min = 0.1*Qc_norm_guess, max=10*Qc_norm_guess)
        self.set_param_hint('dw', min = -1e6, max = 1e6)
        self.set_param_hint('f0', min = f0_guess-3*BW_guess, max=f0_guess+3*BW_guess)
        self.set_param_hint('Qi', expr='1/(1/Ql - 1/Qc_re)') # For convenience, does not affect the fit!
        self.set_param_hint('a', min = 0, max = 100)
        self.set_param_hint('b', min = -1e6, max = 1e6)
        self.set_param_hint('a_nonlin', min = 0, value = 0.01)
        
                       
        params = self.make_params(Ql=Ql_guess, Qc_re=Qc_norm_guess, dw=0, f0=f0_guess, 
                                  a=a_guess, b=b_guess, a_nonlin=1e-3)
        self.guess = lmf.models.update_param_vals(params, self.prefix, **kwargs)

        
## Model from new paper Jonas
