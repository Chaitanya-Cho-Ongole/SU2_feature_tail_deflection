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
nDV = 130
x0 = np.zeros((nDV,))

# Define InputVariable class object: ffd
ffd = InputVariable(0.0, PreStringHandler("DV_VALUE="), nDV)
ffd = InputVariable(x0,ArrayLabelReplacer("__FFD_PTS__"), 0, np.ones(nDV), -0.1,0.1)

# Replace %__DIRECT__% with an empty string when using enable_direct
enable_direct = Parameter([""], LabelReplacer("%__DIRECT__"))

# Replace %__ADJOINT__% with an empty string when using enable_adjoint
enable_adjoint = Parameter([""], LabelReplacer("%__ADJOINT__"))


#enable_not_def = Parameter([""], LabelReplacer("%__NOT_DEF__"))
#enable_def = Parameter([""], LabelReplacer("%__DEF__"))

# Replace *_def.cfg with *_FFD.cfg as required
mesh_in = Parameter(["MESH_FILENAME= CRM_WBT_YP_10_FFD.su2"],\
       LabelReplacer("MESH_FILENAME= CRM_WBT_YP_10_FFD_def.su2"))

# Replace "OBJ_FUNC= SOME NAME" with "OBJ_FUNC= DRAG"
func_drag = Parameter(["OBJECTIVE_FUNCTION= DRAG"],\
         LabelReplacer("OBJECTIVE_FUNCTION= TOPOL_COMPLIANCE"))

# Replace "OBJ_FUNC= SOME NAME" with "OBJ_FUNC= MOMENT_Z"
func_mom = Parameter(["OBJECTIVE_FUNCTION= MOMENT_Y"],\
         LabelReplacer("OBJECTIVE_FUNCTION= TOPOL_COMPLIANCE"))

# EVALUATIONS---------------------------------------#

#Number of of available cores
ncores = "128"

# Master cfg file used for DIRECT and ADJOINT calculations
configMaster="turb_CRM_FBT_FADO.cfg"

deformMaster="deform_CRM_FBT_FADO.cfg"

# cfg file used for WING GEOMETRY calculations
geoMasterWing = "CRM_FBT_geo.cfg"

# Input mesh to perform deformation
meshName="CRM_WBT_YP_10_FFD.su2"

# Mesh deformation
def_command = "mpirun -n " + ncores + " SU2_DEF " + deformMaster

# Geometry evaluation
geo_commandWing = "mpirun -n " + ncores +  " SU2_GEO " + geoMasterWing

# Forward analysis
cfd_command = "mpirun -n " + ncores + " SU2_CFD " + configMaster

# Adjoint analysis -> AD + projection
adjoint_command = "mpirun -n " + ncores + " SU2_CFD_AD " + configMaster + " && mpirun -n " + ncores + " SU2_DOT_AD " + geoMasterWing

# Define the sequential steps for shape-optimization
max_tries = 1

# MESH DEFORMATON------------------------------------------------------#
deform = ExternalRun("DEFORM",def_command,True)
deform.setMaxTries(max_tries)
deform.addConfig(deformMaster)
deform.addData("CRM_WBT_YP_10_FFD.su2")
deform.addExpected("CRM_WBT_YP_10_FFD_def.su2")
deform.addParameter(enable_direct)
deform.addParameter(mesh_in)
#deform.addParameter(enable_def)


# GEOMETRY EVALUATION WING-----------------------------------------#
geometryW = ExternalRun("GEOMETRY_WING",geo_commandWing,True)
geometryW.setMaxTries(max_tries)
geometryW.addConfig(geoMasterWing)
geometryW.addConfig(configMaster)
geometryW.addData("DEFORM/CRM_WBT_YP_10_FFD_def.su2")
geometryW.addExpected("of_func.csv")
geometryW.addExpected("of_grad.csv")

# FORWARD ANALYSIS---------------------------------------------------#
direct = ExternalRun("DIRECT",cfd_command,True)
direct.setMaxTries(max_tries)
direct.addConfig(configMaster)
direct.addData("DEFORM/CRM_WBT_YP_10_FFD_def.su2")
direct.addExpected("solution.dat")
direct.addParameter(enable_direct)


# ADJOINT ANALYSIS---------------------------------------------------#
def makeAdjRun(name, func=None) :
    adj = ExternalRun(name,adjoint_command,True)
    adj.setMaxTries(max_tries)
    adj.addConfig(configMaster)
    adj.addConfig(geoMasterWing)
    adj.addData("DEFORM/CRM_WBT_YP_10_FFD_def.su2")
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


# SCALING PARAMETERS ------------------------------------------------------------ #
GlobalScale = 1
ConScale = 1
FtolCr = 1E-12
Ftol = FtolCr * GlobalScale
OptIter = 0


BSL_WING_VOL = 0.260356    # 40 cores
FACTOR_WV = 1
TRG_WING_VOL = BSL_WING_VOL * FACTOR_WV

# THICKNESS BOUND (xx % of BSL value)
THK_BND = 0.25

# Spanwise thickness values
ST1_T = 0.232115

ST2_T = 0.211534

ST3_T = 0.192532

ST4_T = 0.175896

ST5_T = 0.161056

ST6_T = 0.147767

ST7_T = 0.135624

ST8_T = 0.124602

ST9_T = 0.114473

ST10_T = 0.106967


# DRIVER IPOPT-------------------------------------------------------#
driver = IpoptDriver()


 # Objective function
driver.addObjective("min", drag, GlobalScale)



# Wing pitching moment constraint
driver.addLowerBound(mom, -0.001, GlobalScale , ConScale)

driver.addUpperBound(mom, 0.001, GlobalScale , ConScale)

# Wing volume constraint
#driver.addLowerBound(WingVol, TRG_WING_VOL, GlobalScale)

# Span-wise thickness constraints
#driver.addLowerBound(ST1_TH, ST1_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST2_TH, ST2_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST3_TH, ST3_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST4_TH, ST4_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST5_TH, ST5_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST6_TH, ST6_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST7_TH, ST7_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST8_TH, ST8_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST9_TH, ST9_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST10_TH, ST10_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST11_TH, ST11_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST12_TH, ST12_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST13_TH, ST13_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST14_TH, ST14_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST15_TH, ST15_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST16_TH, ST16_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST17_TH, ST17_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST18_TH, ST18_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST19_TH, ST19_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST20_TH, ST20_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST21_TH, ST21_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST22_TH, ST22_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST23_TH, ST23_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST24_TH, ST24_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST25_TH, ST25_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST26_TH, ST26_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST27_TH, ST27_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST28_TH, ST28_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST29_TH, ST29_T* THK_BND, GlobalScale)
#driver.addLowerBound(ST30_TH, ST30_T* THK_BND, GlobalScale)



driver.setWorkingDirectory("WORKDIR")
driver.setEvaluationMode(False,2.0)

driver.setStorageMode(True,"DSN_")
driver.setFailureMode("HARD")

# DRIVERL IPOPT-------------------------------------------------------#
nlp = driver.getNLP()

# Optimization
x0 = driver.getInitial()

# Warm start parameters
ncon = 2
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
            nlp_scaling_method = "none",    # we are already doing some scaling
            accept_every_trial_step = "yes", # can be used to force single ls per iteration
            limited_memory_max_history = 15,# the "L" in L-BFGS
            max_iter = OptIter,
            tol = Ftol,                     # this and max_iter are the main stopping criteria
            acceptable_iter = OptIter,
            acceptable_tol = Ftol,
            acceptable_obj_change_tol=1e-12, # Cauchy-type convergence over "acceptable_iter"
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

