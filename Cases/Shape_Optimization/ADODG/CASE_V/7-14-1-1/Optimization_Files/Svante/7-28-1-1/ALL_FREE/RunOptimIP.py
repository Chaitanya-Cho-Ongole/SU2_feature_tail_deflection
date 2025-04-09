# Dummy script for FADO_DEV

from FADO import *
import subprocess
import numpy as np
import csv
import os
import glob
import shutil
import pandas as pd
import ipyopt
from numpy import ones, array, zeros

# Design Variables-----#
nDV = 465
x0 = np.zeros((nDV,))

# Define InputVariable class object: ffd
ffd = InputVariable(0.0, PreStringHandler("DV_VALUE="), nDV)
ffd = InputVariable(x0,ArrayLabelReplacer("__FFD_PTS__"), 0, np.ones(nDV), -0.01,0.01)

# Replace %__DIRECT__% with an empty string when using enable_direct
enable_direct = Parameter([""], LabelReplacer("%__DIRECT__"))

# Replace %__ADJOINT__% with an empty string when using enable_adjoint
enable_adjoint = Parameter([""], LabelReplacer("%__ADJOINT__"))


#enable_not_def = Parameter([""], LabelReplacer("%__NOT_DEF__"))
#enable_def = Parameter([""], LabelReplacer("%__DEF__"))

# Replace *_def.cfg with *_FFD.cfg as required
mesh_in = Parameter(["MESH_FILENAME= CRM_WBT_YP_1_M00_FFD.su2"],\
       LabelReplacer("MESH_FILENAME= CRM_WBT_YP_1_M00_FFD_def.su2"))

# Replace "OBJ_FUNC= SOME NAME" with "OBJ_FUNC= DRAG"
func_drag = Parameter(["OBJECTIVE_FUNCTION= DRAG"],\
         LabelReplacer("OBJECTIVE_FUNCTION= TOPOL_COMPLIANCE"))

# Replace "OBJ_FUNC= SOME NAME" with "OBJ_FUNC= MOMENT_Z"
func_mom = Parameter(["OBJECTIVE_FUNCTION= MOMENT_Y"],\
         LabelReplacer("OBJECTIVE_FUNCTION= TOPOL_COMPLIANCE"))

# EVALUATIONS---------------------------------------#

#Number of of available cores
ncores = "48"
ncores2 = "144"

# Master cfg file used for DIRECT and ADJOINT calculations
configMaster="turb_CRM_FBT_FADO.cfg"

deformMaster="deform_CRM_FBT_FADO.cfg"

# cfg file used for WING GEOMETRY calculations
geoMasterWing = "CRM_FBT_geo.cfg"

# Input mesh to perform deformation
meshName="CRM_WBT_YP_1_M00_FFD.su2"

# Mesh deformation
def_command = "mpirun -n " + ncores + " SU2_DEF " + deformMaster

# Geometry evaluation
geo_commandWing = "mpirun -n " + ncores +  " SU2_GEO " + geoMasterWing

# Forward analysis
cfd_command = "mpirun -n " + ncores2 + " --mca sharedfp sm SU2_CFD  " + configMaster

# Adjoint analysis -> AD + projection
adjoint_command = "mpirun -n " + ncores2 + " --mca sharedfp sm SU2_CFD_AD " + configMaster + " && mpirun -n " + ncores2 + " --mca sharedfp sm SU2_DOT_AD " + geoMasterWing

# Define the sequential steps for shape-optimization
max_tries = 1

# MESH DEFORMATON------------------------------------------------------#
deform = ExternalRun("DEFORM",def_command,True)
deform.setMaxTries(max_tries)
deform.addConfig(deformMaster)
deform.addData("CRM_WBT_YP_1_M00_FFD.su2")
deform.addExpected("CRM_WBT_YP_1_M00_FFD_def.su2")
deform.addParameter(enable_direct)
deform.addParameter(mesh_in)
#deform.addParameter(enable_def)


# GEOMETRY EVALUATION WING-----------------------------------------#
geometryW = ExternalRun("GEOMETRY_WING",geo_commandWing,True)
geometryW.setMaxTries(max_tries)
geometryW.addConfig(geoMasterWing)
geometryW.addConfig(configMaster)
geometryW.addData("DEFORM/CRM_WBT_YP_1_M00_FFD_def.su2")
geometryW.addExpected("of_func.csv")
geometryW.addExpected("of_grad.csv")

# FORWARD ANALYSIS---------------------------------------------------#
direct = ExternalRun("DIRECT",cfd_command,True)
direct.setMaxTries(max_tries)
direct.addConfig(configMaster)
direct.addData("DEFORM/CRM_WBT_YP_1_M00_FFD_def.su2")
direct.addExpected("solution.dat")
direct.addParameter(enable_direct)


# ADJOINT ANALYSIS---------------------------------------------------#
def makeAdjRun(name, func=None) :
    adj = ExternalRun(name,adjoint_command,True)
    adj.setMaxTries(max_tries)
    adj.addConfig(configMaster)
    adj.addConfig(geoMasterWing)
    adj.addData("DEFORM/CRM_WBT_YP_1_M00_FFD_def.su2")
    adj.addData("DIRECT/solution.dat")
    adj.addData("DIRECT/flow.meta")
    adj.addExpected("of_grad.dat")
    adj.addParameter(enable_adjoint)
    if (func is not None) : adj.addParameter(func)
    return adj

# Drag adjoint
drag_adj = makeAdjRun("DRAG_ADJ",func_drag)

# Moment adjoint
mom_adj = makeAdjRun("MOM_ADJ",func_mom)

# FUNCTIONS ------------------------------------------------------------ #
drag = Function("drag","DIRECT/history.csv",LabeledTableReader('"CD"'))
drag.addInputVariable(ffd,"DRAG_ADJ/of_grad.dat",TableReader(None,0,(1,0),delim="",zero_except_last=False, zero_last=True))
drag.addValueEvalStep(deform)
drag.addValueEvalStep(direct)
drag.addGradientEvalStep(drag_adj)
drag.setDefaultValue(1.0)

mom = Function("mom","DIRECT/history.csv",LabeledTableReader('"CMy"'))
mom.addInputVariable(ffd,"MOM_ADJ/of_grad.dat",TableReader(None,0,(1,0),delim="",zero_except_last=False, zero_last=False))
mom.addValueEvalStep(deform)
mom.addValueEvalStep(direct)
mom.addGradientEvalStep(mom_adj)
mom.setDefaultValue(-1.0)


WingVol = Function("WingVol","GEOMETRY_WING/of_func.csv",LabeledTableReader('"WING_VOLUME"'))
WingVol.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"WING_VOLUME"',',',(0,None),set_theta_zero=True))
WingVol.addValueEvalStep(deform)
WingVol.addValueEvalStep(geometryW)
WingVol.setDefaultValue(0.0)

ST1_TH = Function("ST1_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION1_THICKNESS"'))
ST1_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION1_THICKNESS"',',',(0,None),set_theta_zero=True))
ST1_TH.addValueEvalStep(deform)
ST1_TH.addValueEvalStep(geometryW)
ST1_TH.setDefaultValue(0.0)

ST2_TH = Function("ST2_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION2_THICKNESS"'))
ST2_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION2_THICKNESS"',',',(0,None),set_theta_zero=True))
ST2_TH.addValueEvalStep(deform)
ST2_TH.addValueEvalStep(geometryW)
ST2_TH.setDefaultValue(0.0)

ST3_TH = Function("ST3_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION3_THICKNESS"'))
ST3_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION3_THICKNESS"',',',(0,None),set_theta_zero=True))
ST3_TH.addValueEvalStep(deform)
ST3_TH.addValueEvalStep(geometryW)
ST3_TH.setDefaultValue(0.0)

ST4_TH = Function("ST4_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION4_THICKNESS"'))
ST4_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION4_THICKNESS"',',',(0,None),set_theta_zero=True))
ST4_TH.addValueEvalStep(deform)
ST4_TH.addValueEvalStep(geometryW)
ST4_TH.setDefaultValue(0.0)

ST5_TH = Function("ST5_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION5_THICKNESS"'))
ST5_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION5_THICKNESS"',',',(0,None),set_theta_zero=True))
ST5_TH.addValueEvalStep(deform)
ST5_TH.addValueEvalStep(geometryW)
ST5_TH.setDefaultValue(0.0)

ST6_TH = Function("ST6_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION6_THICKNESS"'))
ST6_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION6_THICKNESS"',',',(0,None),set_theta_zero=True))
ST6_TH.addValueEvalStep(deform)
ST6_TH.addValueEvalStep(geometryW)
ST6_TH.setDefaultValue(0.0)

ST7_TH = Function("ST7_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION7_THICKNESS"'))
ST7_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION7_THICKNESS"',',',(0,None),set_theta_zero=True))
ST7_TH.addValueEvalStep(deform)
ST7_TH.addValueEvalStep(geometryW)
ST7_TH.setDefaultValue(0.0)

ST8_TH = Function("ST8_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION8_THICKNESS"'))
ST8_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION8_THICKNESS"',',',(0,None),set_theta_zero=True))
ST8_TH.addValueEvalStep(deform)
ST8_TH.addValueEvalStep(geometryW)
ST8_TH.setDefaultValue(0.0)

ST9_TH = Function("ST9_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION9_THICKNESS"'))
ST9_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION9_THICKNESS"',',',(0,None),set_theta_zero=True))
ST9_TH.addValueEvalStep(deform)
ST9_TH.addValueEvalStep(geometryW)
ST9_TH.setDefaultValue(0.0)

ST10_TH = Function("ST10_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION10_THICKNESS"'))
ST10_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION10_THICKNESS"',',',(0,None),set_theta_zero=True))
ST10_TH.addValueEvalStep(deform)
ST10_TH.addValueEvalStep(geometryW)
ST10_TH.setDefaultValue(0.0)

ST11_TH = Function("ST11_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION11_THICKNESS"'))
ST11_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION11_THICKNESS"',',',(0,None),set_theta_zero=True))
ST11_TH.addValueEvalStep(deform)
ST11_TH.addValueEvalStep(geometryW)
ST11_TH.setDefaultValue(0.0)

ST12_TH = Function("ST12_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION12_THICKNESS"'))
ST12_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION12_THICKNESS"',',',(0,None),set_theta_zero=True))
ST12_TH.addValueEvalStep(deform)
ST12_TH.addValueEvalStep(geometryW)
ST12_TH.setDefaultValue(0.0)

ST13_TH = Function("ST13_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION13_THICKNESS"'))
ST13_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION13_THICKNESS"',',',(0,None),set_theta_zero=True))
ST13_TH.addValueEvalStep(deform)
ST13_TH.addValueEvalStep(geometryW)
ST13_TH.setDefaultValue(0.0)

ST14_TH = Function("ST14_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION14_THICKNESS"'))
ST14_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION14_THICKNESS"',',',(0,None),set_theta_zero=True))
ST14_TH.addValueEvalStep(deform)
ST14_TH.addValueEvalStep(geometryW)
ST14_TH.setDefaultValue(0.0)

ST15_TH = Function("ST15_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION15_THICKNESS"'))
ST15_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION15_THICKNESS"',',',(0,None),set_theta_zero=True))
ST15_TH.addValueEvalStep(deform)
ST15_TH.addValueEvalStep(geometryW)
ST15_TH.setDefaultValue(0.0)

ST16_TH = Function("ST16_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION16_THICKNESS"'))
ST16_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION16_THICKNESS"',',',(0,None),set_theta_zero=True))
ST16_TH.addValueEvalStep(deform)
ST16_TH.addValueEvalStep(geometryW)
ST16_TH.setDefaultValue(0.0)

ST17_TH = Function("ST17_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION17_THICKNESS"'))
ST17_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION17_THICKNESS"',',',(0,None),set_theta_zero=True))
ST17_TH.addValueEvalStep(deform)
ST17_TH.addValueEvalStep(geometryW)
ST17_TH.setDefaultValue(0.0)

ST18_TH = Function("ST18_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION18_THICKNESS"'))
ST18_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION18_THICKNESS"',',',(0,None),set_theta_zero=True))
ST18_TH.addValueEvalStep(deform)
ST18_TH.addValueEvalStep(geometryW)
ST18_TH.setDefaultValue(0.0)

ST19_TH = Function("ST19_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION19_THICKNESS"'))
ST19_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION19_THICKNESS"',',',(0,None),set_theta_zero=True))
ST19_TH.addValueEvalStep(deform)
ST19_TH.addValueEvalStep(geometryW)
ST19_TH.setDefaultValue(0.0)

ST20_TH = Function("ST20_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION20_THICKNESS"'))
ST20_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION20_THICKNESS"',',',(0,None),set_theta_zero=True))
ST20_TH.addValueEvalStep(deform)
ST20_TH.addValueEvalStep(geometryW)
ST20_TH.setDefaultValue(0.0)

ST21_TH = Function("ST21_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION21_THICKNESS"'))
ST21_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION21_THICKNESS"',',',(0,None),set_theta_zero=True))
ST21_TH.addValueEvalStep(deform)
ST21_TH.addValueEvalStep(geometryW)
ST21_TH.setDefaultValue(0.0)

ST22_TH = Function("ST22_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION22_THICKNESS"'))
ST22_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION22_THICKNESS"',',',(0,None),set_theta_zero=True))
ST22_TH.addValueEvalStep(deform)
ST22_TH.addValueEvalStep(geometryW)
ST22_TH.setDefaultValue(0.0)

ST23_TH = Function("ST23_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION23_THICKNESS"'))
ST23_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION23_THICKNESS"',',',(0,None),set_theta_zero=True))
ST23_TH.addValueEvalStep(deform)
ST23_TH.addValueEvalStep(geometryW)
ST23_TH.setDefaultValue(0.0)

ST24_TH = Function("ST24_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION24_THICKNESS"'))
ST24_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION24_THICKNESS"',',',(0,None),set_theta_zero=True))
ST24_TH.addValueEvalStep(deform)
ST24_TH.addValueEvalStep(geometryW)
ST24_TH.setDefaultValue(0.0)

ST25_TH = Function("ST25_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION25_THICKNESS"'))
ST25_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION25_THICKNESS"',',',(0,None),set_theta_zero=True))
ST25_TH.addValueEvalStep(deform)
ST25_TH.addValueEvalStep(geometryW)
ST25_TH.setDefaultValue(0.0)

ST26_TH = Function("ST26_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION26_THICKNESS"'))
ST26_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION26_THICKNESS"',',',(0,None),set_theta_zero=True))
ST26_TH.addValueEvalStep(deform)
ST26_TH.addValueEvalStep(geometryW)
ST26_TH.setDefaultValue(0.0)

ST27_TH = Function("ST27_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION27_THICKNESS"'))
ST27_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION27_THICKNESS"',',',(0,None),set_theta_zero=True))
ST27_TH.addValueEvalStep(deform)
ST27_TH.addValueEvalStep(geometryW)
ST27_TH.setDefaultValue(0.0)

ST28_TH = Function("ST28_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION28_THICKNESS"'))
ST28_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION28_THICKNESS"',',',(0,None),set_theta_zero=True))
ST28_TH.addValueEvalStep(deform)
ST28_TH.addValueEvalStep(geometryW)
ST28_TH.setDefaultValue(0.0)

ST29_TH = Function("ST29_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION29_THICKNESS"'))
ST29_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION29_THICKNESS"',',',(0,None),set_theta_zero=True))
ST29_TH.addValueEvalStep(deform)
ST29_TH.addValueEvalStep(geometryW)
ST29_TH.setDefaultValue(0.0)

ST30_TH = Function("ST30_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION30_THICKNESS"'))
ST30_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION30_THICKNESS"',',',(0,None),set_theta_zero=True))
ST30_TH.addValueEvalStep(deform)
ST30_TH.addValueEvalStep(geometryW)
ST30_TH.setDefaultValue(0.0)


# SCALING PARAMETERS ------------------------------------------------------------ #
GlobalScale = 1
ConScale = 1
FtolCr = 1E-12
Ftol = FtolCr * GlobalScale
OptIter = 50


BSL_WING_VOL = 88.8998    # 48 cores
FACTOR_WV = 0.9999
TRG_WING_VOL = BSL_WING_VOL * FACTOR_WV

# THICKNESS BOUND (xx % of BSL value)
THK_BND = 0.99

# Spanwise thickness values
ST1_T = 1.6194

ST2_T = 1.4754

ST3_T = 1.34361

ST4_T = 1.22805

ST5_T = 1.12503

ST6_T = 1.03228

ST7_T = 0.947909

ST8_T = 0.871203

ST9_T = 0.80104

ST10_T = 0.748752

ST11_T = 0.716289

ST12_T = 0.686041

ST13_T = 0.655433

ST14_T = 0.62615

ST15_T = 0.599399

ST16_T = 0.573919

ST17_T = 0.548387

ST18_T = 0.523925

ST19_T = 0.499889

ST20_T = 0.475987

ST21_T = 0.453178

ST22_T = 0.430781

ST23_T = 0.409025

ST24_T = 0.386961

ST25_T = 0.365446

ST26_T = 0.344085

ST27_T = 0.32286

ST28_T = 0.301635

ST29_T = 0.280385

ST30_T = 0.24893






# DRIVER IPOPT-------------------------------------------------------#
driver = IpoptDriver()


 # Objective function
driver.addObjective("min", drag, GlobalScale)



# Wing pitching moment constraint
#driver.addLowerBound(mom, -0.00001, GlobalScale , ConScale)

#driver.addUpperBound(mom, 0.00001, GlobalScale , ConScale)

# Wing volume constraint
driver.addLowerBound(WingVol, TRG_WING_VOL, GlobalScale)

# Span-wise thickness constraints
driver.addLowerBound(ST1_TH, ST1_T* THK_BND, GlobalScale)
driver.addLowerBound(ST2_TH, ST2_T* THK_BND, GlobalScale)
driver.addLowerBound(ST3_TH, ST3_T* THK_BND, GlobalScale)
driver.addLowerBound(ST4_TH, ST4_T* THK_BND, GlobalScale)
driver.addLowerBound(ST5_TH, ST5_T* THK_BND, GlobalScale)
driver.addLowerBound(ST6_TH, ST6_T* THK_BND, GlobalScale)
driver.addLowerBound(ST7_TH, ST7_T* THK_BND, GlobalScale)
driver.addLowerBound(ST8_TH, ST8_T* THK_BND, GlobalScale)
driver.addLowerBound(ST9_TH, ST9_T* THK_BND, GlobalScale)
driver.addLowerBound(ST10_TH, ST10_T* THK_BND, GlobalScale)
driver.addLowerBound(ST11_TH, ST11_T* THK_BND, GlobalScale)
driver.addLowerBound(ST12_TH, ST12_T* THK_BND, GlobalScale)
driver.addLowerBound(ST13_TH, ST13_T* THK_BND, GlobalScale)
driver.addLowerBound(ST14_TH, ST14_T* THK_BND, GlobalScale)
driver.addLowerBound(ST15_TH, ST15_T* THK_BND, GlobalScale)
driver.addLowerBound(ST16_TH, ST16_T* THK_BND, GlobalScale)
driver.addLowerBound(ST17_TH, ST17_T* THK_BND, GlobalScale)
driver.addLowerBound(ST18_TH, ST18_T* THK_BND, GlobalScale)
driver.addLowerBound(ST19_TH, ST19_T* THK_BND, GlobalScale)
driver.addLowerBound(ST20_TH, ST20_T* THK_BND, GlobalScale)
driver.addLowerBound(ST21_TH, ST21_T* THK_BND, GlobalScale)
driver.addLowerBound(ST22_TH, ST22_T* THK_BND, GlobalScale)
driver.addLowerBound(ST23_TH, ST23_T* THK_BND, GlobalScale)
driver.addLowerBound(ST24_TH, ST24_T* THK_BND, GlobalScale)
driver.addLowerBound(ST25_TH, ST25_T* THK_BND, GlobalScale)
driver.addLowerBound(ST26_TH, ST26_T* THK_BND, GlobalScale)
driver.addLowerBound(ST27_TH, ST27_T* THK_BND, GlobalScale)
driver.addLowerBound(ST28_TH, ST28_T* THK_BND, GlobalScale)
driver.addLowerBound(ST29_TH, ST29_T* THK_BND, GlobalScale)
driver.addLowerBound(ST30_TH, ST30_T* THK_BND, GlobalScale)



driver.setWorkingDirectory("WORKDIR")
driver.setEvaluationMode(False,2.0)

driver.setStorageMode(True,"DSN_")
driver.setFailureMode("HARD")

# DRIVERL IPOPT-------------------------------------------------------#
nlp = driver.getNLP()

# Optimization
x0 = driver.getInitial()

# Warm start parameters
ncon = 31
lbMult = np.zeros(nDV)
ubMult= np.zeros(nDV)
conMult = np.zeros(ncon)

print("Initial Design Variable vector:")
print(x0)

# WARM START PARAMETERS
#x0 = array([])

#ubMult = array([])

#lbMult = array([])

#conMult = array([])


# NLP settings
nlp.set(warm_start_init_point = "no",
            nlp_scaling_method = "gradient-based",
            nlp_scaling_max_gradient = 1.0,
            accept_every_trial_step = "no", # can be used to force single ls per iteration
            limited_memory_max_history = 50,
            max_iter = OptIter,
            tol = Ftol,
            mu_strategy = "adaptive",
            mu_oracle = "quality-function",
            adaptive_mu_globalization="obj-constr-filter",
            sigma_max = 100.0,
            sigma_min = 1e-06,
            acceptable_iter = OptIter,
            acceptable_tol = Ftol,
            acceptable_obj_change_tol=1e-5, # Cauchy-type convergence over "acceptable_iter"
            dual_inf_tol=1e-07,             # Tolerance for optimality criteria
            mu_min = 1e-8,                  # for very low values (e-10) the problem "flip-flops"
            recalc_y_feas_tol = 0.1,
            output_file = 'ipopt_output.txt')        # helps converging the dual problem with L-BFGS

x, obj, status = nlp.solve(x0, mult_g = conMult, mult_x_L = lbMult, mult_x_U = ubMult)
#x, obj, status = nlp.solve(x0)
# Print the optimized results---->

print("Primal variables solution")
print("x: ", x)

print("Bound multipliers solution: Lower bound")
print("lbMult: ", lbMult)

print("Bound multipliers solution: Upper bound")
print("ubMult: ", ubMult)


#print("Constraint multipliers solution")
print("lambda:",conMult)

