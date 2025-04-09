import subprocess
import os
import shutil
import glob
import pandas as pd
import os
import numpy as np

def process_su2_history(history_file, global_csv, mach_number):
    """
    Reads the SU2-generated history.csv file, extracts the last values of CD, CL, CMx, CMy, and CMz,
    and appends these values along with the corresponding Mach number to a global CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(history_file)
        
        # Clean column names (strip spaces and remove quotes)
        df.columns = df.columns.str.strip().str.replace('"', '')

        # Ensure required columns exist
        required_columns = ["CD", "CL", "CMx", "CMy", "CMz", "AoA"]
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Missing required columns in {history_file}")
            return

        # Extract the last row values
        last_values = df.iloc[-1][required_columns].to_dict()
        last_values["Mach"] = mach_number  # Add Mach number

        # Convert dictionary to DataFrame
        result_df = pd.DataFrame([last_values])

        # Append to global CSV file, creating if necessary
        file_exists = os.path.isfile(global_csv)
        result_df.to_csv(global_csv, mode='a', header=not file_exists, index=False)

        #print(f"Appended results from {history_file} to {global_csv}")
    
    except Exception as e:
        print(f"Error processing {history_file}: {e}")


# Remove "Results" directory if it exists
results_dir = "Results"
if os.path.exists(results_dir):
    #print(f"Removing existing '{results_dir}' directory...")
    shutil.rmtree(results_dir)
    
global_csv_path = "Polar_results.csv"  # Define global CSV file path
    
# Remove global_results.csv if it exists
if os.path.exists(global_csv_path):
    #print(f"Removing existing '{global_csv_path}' file...")
    os.remove(global_csv_path)


def modify_su2_config(input_file, output_file, TARGET_AOA, MACH, restart=False):
    """ Modifies an SU2 configuration file by setting FIXED_CL_MODE to YES, TARGET_AOA, and RESTART_SOL. """
    with open(input_file, "r") as file:
        lines = file.readlines()

    with open(output_file, "w") as file:
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("%"):
                file.write(line)
                continue

            if "=" in stripped_line:
                key, value = map(str.strip, stripped_line.split("=", 1))
                if key == "MACH_NUMBER":
                    line = f"{key}= {MACH}\n"
                elif key == "AOA":
                    line = f"{key}= {TARGET_AOA}\n"
                #elif key == "RESTART_SOL":
                #    line = f"{key}= {'YES' if restart else 'NO'}\n"  # First CL = NO, others = YES
                
            file.write(line)

    return output_file

def update_restart_config(input_file):
    """
    Updates RESTART_SOL=YES in the SU2 configuration file.
    """
    temp_file = input_file + ".tmp"
    with open(input_file, "r") as file, open(temp_file, "w") as temp:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("%"):
                temp.write(line)
                continue

            if "=" in stripped_line:
                key, value = map(str.strip, stripped_line.split("=", 1))
                if key == "RESTART_SOL":
                    line = f"{key}= YES\n"  # Enable restart for the next CL
                
            temp.write(line)

    os.replace(temp_file, input_file)  # Overwrite the original file
    #print(f"Updated restart configuration in {input_file}")

def run_su2_simulation(num_cores, cfg_file, TARGET_AOA, MACH, restart=False, next_cl_dir=None):
    """
    Runs SU2_CFD, logs results, and prepares the next AOA case with RESTART_SOL=YES.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Create Mach-specific directory
    mach_dir = os.path.join(results_dir, f"MACH_{MACH:.2f}".replace(".", "_"))
    os.makedirs(mach_dir, exist_ok=True)

    # Create CL-specific directory
    work_dir = os.path.join(mach_dir, f"AOA_{TARGET_AOA:.2f}".replace(".", "_"))
    os.makedirs(work_dir, exist_ok=True)

    log_file_path = os.path.join(work_dir, "simulation.log")

    # Copy config file
    cfg_dest = os.path.join(work_dir, os.path.basename(cfg_file))
    shutil.copy(cfg_file, cfg_dest)

    # Create symbolic links for .su2 files
    for su2_file in glob.glob("*.su2"):
        link_dest = os.path.join(work_dir, su2_file)
        if not os.path.exists(link_dest):
            os.symlink(os.path.abspath(su2_file), link_dest)
    
    # Modify configuration file
    modified_cfg = os.path.join(work_dir, "modified_" + os.path.basename(cfg_file))
    modify_su2_config(cfg_dest, modified_cfg, TARGET_AOA, MACH, restart)

    # Run SU2_CFD with mpirun
    command = ["mpirun", "-np", str(num_cores), "SU2_CFD", os.path.basename(modified_cfg)]
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            command, cwd=work_dir, stdout=log_file, stderr=log_file, text=True
        )
        process.wait()
    
    history_file = os.path.join(work_dir, "history.csv")
    process_su2_history(history_file, global_csv_path, MACH)

    # Move solution.dat to the next CL directory if applicable
    solution_file = os.path.join(work_dir, "solution.dat")
    flow_meta_file = os.path.join(work_dir, "flow.meta")
    if next_cl_dir:
        os.makedirs(next_cl_dir, exist_ok=True)

        # Move solution.dat
        if os.path.exists(solution_file):
            #print(f"Moving 'solution.dat' to {next_cl_dir}")
            shutil.move(solution_file, os.path.join(next_cl_dir, "solution.dat"))

        # Move flow.meta
        if os.path.exists(flow_meta_file):
            #print(f"Moving 'flow.meta' to {next_cl_dir}")
            shutil.move(flow_meta_file, os.path.join(next_cl_dir, "flow.meta"))
            
        # Copy modified configuration file to the next CL directory
        next_cfg = os.path.join(next_cl_dir, "modified_" + os.path.basename(cfg_file))
        shutil.copy(modified_cfg, next_cfg)

        # Update RESTART_SOL=YES in the next CL directory before the next simulation
        #update_restart_config(next_cfg)

    return log_file_path

# Example usage:
ncores = 96
cfg_file = "turb_CRM_FBT_FADO.cfg"

# Define step sizes
mach_step = 0.05  # Change this to adjust Mach increments
aoa_step = 0.01  # Change this to adjust CL increments

# Define Mach and CL values for nested loops
MACH_VALUES = np.arange(0.50, 0.85 + mach_step, mach_step)  # Mach from 0.50 to 0.85 in steps of n
AOA_VALUES = np.arange(-2.0, 3.0 + aoa_step, aoa_step)  # CL from 0.0 to 0.7 in steps of n

# Run simulations for all (MACH, CL) combinations
for i, MACH in enumerate(MACH_VALUES):
    for j, TARGET_AOA in enumerate(AOA_VALUES):
        print(f"Running SU2_CFD for Mach={MACH}, AOA={TARGET_AOA} ...", end=" ", flush=True)

        next_cl_dir = None
        restart = j > 0  # Restart from the second CL onwards

        if j < len(AOA_VALUES) - 1:
            next_cl_dir = os.path.join(
                "Results", f"MACH_{MACH:.2f}".replace(".", "_"), f"CL_{AOA_VALUES[j+1]:.2f}".replace(".", "_")
            )

        log_path = run_su2_simulation(ncores, cfg_file, TARGET_AOA, MACH, restart, next_cl_dir)
        print("Done!")
    print("\n")
