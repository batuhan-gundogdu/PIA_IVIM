import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import time
from tqdm import tqdm
from scipy.optimize import curve_fit








def detect_PIDS_slice(b, S):
    """ Inputs: b - diffusion weight values used in image
                S - Hybrid Multi-dimensional image
        Outputs:
                PIDS_ADC1 : Binary Map with voxels ADC > 3 (could mean motion induced signal loss at high-b)
                PIDS_ADC2 : Binary Map with voxels ADC < 0 (could mean the voxel is below the noise level)
                PIDS_b_decay : Binary Map with voxels disobeying decay rule along b direction
                PIDS_TE_decay : Binary Map with voxels disobeying decay rule along TE direction
    """
    
    eps = 1e-7
    localize = np.eye(4)
    num_rows, num_cols, num_bvalues, num_TEs = S.shape
    PIDS_ADC1 = np.zeros((num_rows, num_cols))
    PIDS_ADC2 = np.zeros((num_rows, num_cols))
    PIDS_b_decay = np.zeros((num_rows, num_cols, num_TEs, 3))
    PIDS_TE_decay = np.zeros((num_rows, num_cols, num_bvalues, 3))
    for row in tqdm(range(num_rows)):
         for col in range(num_cols):
            te0 = np.squeeze(S[row,col, :, 0])
            adc = np.polyfit(b.flatten()/1000, np.log(te0 + eps), 1)
            adc = -adc[0]    
            PIDS_ADC1[row, col] = int(adc > 3)
            PIDS_ADC2[row, col] = int(adc < 0)
            for _b in range(num_bvalues):
                signals_along_te = np.squeeze(S[row,col, _b, :])
                to_compare = signals_along_te.copy().astype(int)
                to_compare[1:] = signals_along_te[:3]
                is_pids = signals_along_te - to_compare
                for local in range(3):
                    is_pids_ = int(is_pids[local + 1]>=0)
                    PIDS_TE_decay[row, col, _b, local] = is_pids_
            for _te in range(num_TEs):
                signals_along_b = np.squeeze(S[row,col, :, _te])
                to_compare = signals_along_b.copy().astype(int)
                to_compare[1:] = signals_along_b[:3]
                is_pids = signals_along_b - to_compare
                for local in range(3):
                    is_pids_ = int(is_pids[local + 1]>=0)
                    PIDS_b_decay[row, col, _te, local] = is_pids_

    return PIDS_ADC1, PIDS_ADC2, PIDS_b_decay, PIDS_TE_decay

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'