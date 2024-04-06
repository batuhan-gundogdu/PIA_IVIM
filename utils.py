import numpy as np
from scipy.optimize import curve_fit
import torch
from tqdm import tqdm

def read_data(file_dir, fname, i):
    
    fname_tmp = file_dir + "{:04}".format(i) + fname
    data = np.load(fname_tmp)
    
    return data


def rRMSE(x,y,t, is_f=False):
    
    Nx, Ny = x.shape

    t_tmp = np.reshape(t, (Nx*Ny,))
    tumor_indice = np.argwhere(t_tmp == 8)
    non_tumor_indice = np.argwhere(t_tmp != 8)
    non_air_indice = np.argwhere(t_tmp != 1)
    non_tumor_air_indice= np.intersect1d(non_tumor_indice,non_air_indice)
    
    x_tmp = np.reshape(x, (Nx*Ny,))
    x_t = x_tmp[tumor_indice]
    x_nt = x_tmp[non_tumor_air_indice]
    
    y_tmp = np.reshape(y, (Nx*Ny,))
    y_t = y_tmp[tumor_indice]
    y_nt = y_tmp[non_tumor_air_indice]
    
    # tumor region
    tmp1 = np.sqrt(tumor_indice.shape[0]) if is_f else np.sqrt(np.sum(np.square(y_t)))
    tmp2 = np.sqrt(np.sum(np.square(x_t-y_t)))
    z_t = tmp2 / tmp1
    
    # non-tumor region
    tmp1 = np.sqrt(non_tumor_air_indice.shape[0]) if is_f else np.sqrt(np.sum(np.square(y_nt)))
    tmp2 = np.sqrt(np.sum(np.square(x_nt-y_nt)))
    z_nt = tmp2 / tmp1
    
    return z_t, z_nt

def rRMSE_per_case(x_f,x_dt,x_ds,y_f,y_dt,y_ds,t):
    
    
    R_f_t, R_f_nt = rRMSE(x_f, y_f, t, is_f=True)
    R_Dt_t, R_Dt_nt = rRMSE(x_dt, y_dt, t)
    R_Ds_t, R_Ds_nt = rRMSE(x_ds, y_ds, t)
    
    z =  (R_f_t + R_Dt_t + R_Ds_t)/3 + (R_f_nt + R_Dt_nt)/2
    
    z_t =  (R_f_t + R_Dt_t + R_Ds_t)/3
    
    return z, z_t


def rRMSE_all_cases(x_f,x_dt,x_ds,y_f,y_dt,y_ds,t):
    
    z = np.empty([x_f.shape[2]])
    z_t = np.empty([x_f.shape[2]])
    
    for i in range(x_f.shape[2]):
        z[i], z_t[i] = rRMSE_per_case(x_f[:,:,i],x_dt[:,:,i],x_ds[:,:,i],y_f[:,:,i],y_dt[:,:,i],y_ds[:,:,i],t[:,:,i]) 
        
    return np.average(z), np.average(z_t)

def funcBiExp(b, f, Dt, Ds):
    ## Units
    # b: s/mm^2
    # D: mm^2/s
    return (1.-f) * np.exp(-1.*Dt * (b/1000)) + f * np.exp(-1.*Ds * (b/1000))
 
def fit_biExponential_model(arr3D_img, arr1D_b):
    

    arr2D_coordBody = np.argwhere(arr3D_img[:,:,0]>0)
    arr2D_fFitted = np.zeros_like(arr3D_img[:,:,0])
    arr2D_DtFitted = np.zeros_like(arr3D_img[:,:,0])
    arr2D_DsFitted = np.zeros_like(arr3D_img[:,:,0])

    for arr1D_coord in arr2D_coordBody:
        try:
            popt, pcov = curve_fit(funcBiExp, arr1D_b[1:]-arr1D_b[0], arr3D_img[arr1D_coord[0],arr1D_coord[1],1:]/arr3D_img[arr1D_coord[0],arr1D_coord[1],0]
                                , p0=(0.15,1.5e-3,8e-3), bounds=([0, 0, 3.0e-3], [1, 2.9e-3, np.inf]), method='trf')

        except:
            popt = [0, 0, 0]
            print('Coord {} fail to be fitted, set all parameters as 0'.format(arr1D_coord))

        arr2D_fFitted[arr1D_coord[0], arr1D_coord[1]] = popt[0]
        arr2D_DtFitted[arr1D_coord[0], arr1D_coord[1]] = popt[1]
        arr2D_DsFitted[arr1D_coord[0], arr1D_coord[1]] = popt[2]

    return np.concatenate((arr2D_fFitted[:,:,np.newaxis],arr2D_DtFitted[:,:,np.newaxis],arr2D_DsFitted[:,:,np.newaxis]), axis=2)



def fit_biExponential_model_signal(signal, b):
    
    numcols, acquisitions = signal.shape
    f = np.zeros((numcols,))
    D = np.zeros((numcols,))
    D_star = np.zeros((numcols,))
    for col in tqdm(range(numcols)):
        xdata = b
        ydata = signal[col]/signal[col,0]
        try:
            popt, pcov = curve_fit(funcBiExp, xdata, ydata
                                , p0=(0.15, 1.5, 8), bounds=([0, 0, 3.0], [1, 2.9, np.inf]), method='trf')

        except:
            popt = [0, 0, 0]
            print('fail to be fitted, set all parameters as 0')

        f[col] = popt[0]
        D[col] = popt[1]
        D_star[col] = popt[2]

    return f, D, D_star

    



def get_batch(batch_size=16, noise_sdt=0.01):

    b_values = [0, 5, 50, 100, 200, 500, 800, 1000]

    
    D = np.random.uniform(0., 3., batch_size)
    D_star = np.random.uniform(3.0, 60., batch_size)    
    f = np.random.uniform(0, 1, batch_size)
    
    signal = np.zeros((batch_size, len(b_values)), dtype=float)
    for sample in range(batch_size):
        for ctr, b in enumerate(b_values):
            signal[sample, ctr] = (1 - f[sample])*np.exp(-(b/1000)*D[sample]) + f[sample]*np.exp(-(b/1000)*D_star[sample])


    noise_im = np.random.normal(0, noise_sdt, signal.shape)
    noise_re = np.random.normal(0, noise_sdt, signal.shape)
    noisy = np.sqrt((signal + noise_im)**2 + (signal + noise_re)**2)/np.sqrt(2)

    return torch.from_numpy(noisy).float(), torch.from_numpy(f.T).float(), torch.from_numpy(D.T).float(), torch.from_numpy(D_star.T).float(),  torch.from_numpy(signal).float()
