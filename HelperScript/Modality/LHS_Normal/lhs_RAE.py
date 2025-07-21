import os
import numpy as np
import shutil
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm

"""
Script: sample_deformed_crm.py

Description:
------------
This script performs Latin Hypercube Sampling (LHS) of 320 design variables (DVs) for a
Common Research Model (CRM) airfoil/wing deformation study using SU2.

Each sample perturbs the default DV vector (initialized as zeros) using a normal distribution 
centered at 0.0 with a user-specified standard deviation (`Magnitude`). The script:

1. Generates N sample DV vectors using LHS and maps them to a normal distribution.
2. Creates a new directory for each sample.
3. Copies the baseline SU2 configuration and mesh files into each directory.
4. Replaces the DV line in the deformation config with the sampled DV vector.
5. Runs SU2_DEF to perform geometry deformation.
6. Runs SU2_CFD to perform CFD analysis.

Parameters:
-----------
- DVs : np.array
    Default design variable vector (zeros).
- N : int
    Number of Latin Hypercube samples to generate.
- Magnitude : float
    Standard deviation of the normal distribution for DV perturbation.
- CoreN : int
    Number of MPI cores used to run SU2_DEF and SU2_CFD.
- Origin : str
    Directory containing the baseline configuration and mesh files.
- DV line index (hardcoded): int
    The DV line in the SU2 config file is assumed to be line 388 (index 387).

Requirements:
-------------
- SU2 installed and available in PATH with MPI support.
- `scipy`, `numpy`, `shutil`, and `os` Python modules.
- The baseline folder (`Origin`) must contain:
    - def_CRM_FBT_FADO.cfg
    - turb_CRM_FBT_FADO.cfg
    - CRM_WBT_YP_1_M00_FFD.su2

Outputs:
--------
- For each of the N samples, a folder `Deformed_CRM_WBT_i` is created.
- Each folder contains:
    - Modified SU2 config files
    - Mesh file
    - `def.out` (output from SU2_DEF)
    - `sol.out` (output from SU2_CFD)

Notes:
------
- The script assumes that `DV_VALUE=` is located on line 388 of the deformation config file.
- LHS ensures stratified sampling in all DV dimensions.
- Normal distribution is used to control perturbation magnitude in a statistically meaningful way.

"""


# Specify the location of optimized wing DVs
DVs = np.zeros(320)  # Default values are 0.0

# Specify location of original simulations
Origin = 'DV_64'

# Specify number of cores to use
CoreN = 240

# Number of samples and magnitude of normal variation (std dev)
N = 20
Magnitude = 0.05  # Standard deviation for N(0.0, sigma^2)

# Create LHS sampler
engine = LatinHypercube(d=len(DVs), seed=42)
lhs_uniform = engine.random(N)

# Transform LHS uniform samples to normal distribution centered at 0.0
Samples = norm(loc=0.0, scale=Magnitude).ppf(lhs_uniform)

for i in range(N):
    Path = f'Deformed_CRM_WBT_{i+1}'
    os.mkdir(Path)

    # Copy configuration and mesh files into the new sample directory
    cfgO = f'{Origin}/def_CRM_FBT_FADO.cfg'
    cfgD = f'{Path}/def_CRM_FBT_FADO.cfg'
    cfg1O = f'{Origin}/turb_CRM_FBT_FADO.cfg'
    cfg1D = f'{Path}/turb_CRM_FBT_FADO.cfg'
    meshO = f'{Origin}/CRM_WBT_YP_1_M00_FFD.su2'
    meshD = f'{Path}/CRM_WBT_YP_1_M00_FFD.su2'

    shutil.copy2(cfgO, cfgD)
    shutil.copy2(cfg1O, cfg1D)
    shutil.copy2(meshO, meshD)

    os.chdir(Path)

    # Apply the i-th sample as a perturbation around the default DVs
    CurrSam = DVs + Samples[i]
    converted_list = [str(element) for element in CurrSam]
    joined_string = ", ".join(converted_list)
    DVNew = 'DV_VALUE= ' + joined_string + '\n'

    # Modify the CFG file in-place at the DV_VALUE line
    with open('def_CRM_FBT_FADO.cfg', 'r') as file:
        data = file.readlines()

    data[387] = DVNew  # ⚠ Assumes DV_VALUE is always on line 388

    with open('def_CRM_FBT_FADO.cfg', 'w') as file:
        file.writelines(data)

    # Run deformation
    os.system(f'mpirun -n {CoreN} SU2_DEF def_CRM_FBT_FADO.cfg > def.out')
    print('Deformation Complete for ' + Path)

    # Run CFD analysis
    os.system(f'mpirun -n {CoreN} SU2_CFD turb_CRM_FBT_FADO.cfg > sol.out')

    os.chdir('../')
