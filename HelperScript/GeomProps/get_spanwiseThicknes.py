import pandas as pd

def read_and_print_thickness(file_path):
    """
    Reads the CSV file and prints the thickness values for STATION1_THICKNESS to STATION30_THICKNESS.
    """
    df = pd.read_csv(file_path)
    
    # Filter columns that match the pattern "STATION#_THICKNESS"
    thickness_columns = [col for col in df.columns if "STATION" in col and "THICKNESS" in col]
    
    # Sort columns numerically based on station number
    sorted_columns = sorted(thickness_columns, key=lambda x: int(x.split("STATION")[1].split("_")[0]))

    # Print the relevant thickness values
    for col in sorted_columns:
        print(f"{col}: {df[col].values[0]}")

# Example usage with file path
file_path = "of_func.csv"
read_and_print_thickness(file_path)
