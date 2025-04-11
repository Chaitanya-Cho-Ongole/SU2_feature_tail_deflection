import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_surface_normal_at_slice(df, y_value=20, epsilon=1e-2):
    """
    Computes the surface normal at Y = y_value using the cross product
    of tangent vectors in the X-Z plane from a set of 3 spatially distinct points.
    Flips the normal to point in the +Y direction.
    """
    df_slice = df[np.abs(df['Y'] - y_value) < epsilon].reset_index(drop=True)
    if len(df_slice) < 3:
        raise ValueError("Not enough points in the Y-slice to compute a normal.")

    p0 = df_slice.iloc[0][['X', 'Y', 'Z']].values
    p1 = df_slice.iloc[1][['X', 'Y', 'Z']].values
    p2 = df_slice.iloc[2][['X', 'Y', 'Z']].values

    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)

    if normal[1] < 0:
        normal = -normal

    normal_unit = normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else np.zeros(3)
    return p0, normal_unit

def compute_multiple_normals(df, y_range=(2, 28), num_points=10, epsilon=1e-2):
    y_values = np.linspace(y_range[0], y_range[1], num_points)
    normals_list = []

    for y in y_values:
        try:
            base, normal = compute_surface_normal_at_slice(df, y_value=y, epsilon=epsilon)
            normals_list.append((base, normal))
        except ValueError:
            print(f"Not enough points near Y = {y:.2f} to compute a normal.")
            continue

    return y_values, normals_list

def plot_translucent_mesh_with_multiple_normals_downsampled(df, normals_list, alpha=0.01, dpi=300, max_points=100000):
    """
    Plots a downsampled 3D mesh and overlays multiple surface normal vectors.
    """
    if len(df) > max_points:
        df_plot = df
    else:
        df_plot = df

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df_plot['X'], df_plot['Y'], df_plot['Z'], c='gray', alpha=alpha, s=0.01)

    for base_point, normal in normals_list:
        ax.quiver(base_point[0], base_point[1], base_point[2],
                  normal[0], normal[1], normal[2],
                  length=2.0, color='black', normalize=True, linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title("Translucent Downsampled Mesh with Y+ Surface Normals")

    plt.show()

# --- Execution Block ---

# Load your CSV file
df = pd.read_csv("candidate_points.csv")  # Update path if needed

# Compute and visualize Y+ normals
y_vals, multi_normals = compute_multiple_normals(df, y_range=(3, 28), num_points=10)
plot_translucent_mesh_with_multiple_normals_downsampled(df, multi_normals)

# Print results
for i, (base, normal) in enumerate(multi_normals):
    print(f"Y ≈ {base[1]:.2f}: Base Point = {base}, Normal = {normal}")


