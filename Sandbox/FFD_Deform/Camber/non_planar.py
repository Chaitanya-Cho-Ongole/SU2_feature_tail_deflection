import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import UnivariateSpline

def load_data(surface_path, slice_path):
    surface_df = pd.read_csv(surface_path)
    slice_df = pd.read_csv(slice_path)
    slice_df.columns = slice_df.columns.str.strip()
    y_slices = slice_df['Location'].astype(float).values
    return surface_df, y_slices

def extract_max_z(surface_df, y_slices, delta_y=0.01):
    max_z_points = []
    for y_target in y_slices:
        slice_points = surface_df[np.abs(surface_df['Y'] - y_target) < delta_y]
        if not slice_points.empty:
            max_idx = slice_points['Z'].idxmax()
            max_point = surface_df.loc[max_idx]
            max_z_points.append((max_point['X'], max_point['Y'], max_point['Z']))
    return np.array(max_z_points)

def plot_3d_surface_with_slices(surface_df, y_slices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(surface_df['X'], surface_df['Y'], surface_df['Z'], c='gray', alpha=0.05, s=1.5)

    x_min, x_max = surface_df['X'].min(), surface_df['X'].max()
    z_min, z_max = surface_df['Z'].min(), surface_df['Z'].max()

    for y in y_slices:
        X_plane, Z_plane = np.meshgrid([x_min, x_max], [z_min, z_max])
        Y_plane = np.full_like(X_plane, y)
        ax.plot_surface(X_plane, Y_plane, Z_plane, color='C0', alpha=0.05, edgecolor='none')

    ax.set_xlabel('X', fontsize=22, fontname="Times New Roman")
    ax.set_ylabel('Y', fontsize=22, fontname="Times New Roman")
    ax.set_zlabel('Z', fontsize=22, fontname="Times New Roman")
    ax.grid(False)
    ax.set_axis_off()
    
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0]*1.5, Size[1]*1.5, forward=True)
    
    
    plt.tight_layout()
    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('Plots/non_planar_FFD.png')
    
def plot_xy_projection_with_slices(surface_df, y_slices):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot in X-Y plane (ignoring Z)
    ax.scatter(surface_df['X'], surface_df['Y'], c='gray', alpha=0.05, s=1.5)

    # Plot horizontal lines at each y-slice location
    x_min, x_max = surface_df['X'].min(), surface_df['X'].max()
    for y in y_slices:
        ax.plot([x_min, x_max], [y, y], color='blue', alpha=0.3, linestyle='--')

    ax.set_xlabel('X', fontsize=18, fontname="Times New Roman")
    ax.set_ylabel('Y', fontsize=18, fontname="Times New Roman")
    ax.grid(True)
    ax.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.show()

def plot_yz_projection_with_vectors(max_z_points):
    from scipy.interpolate import splprep, splev

    fig, ax = plt.subplots()
    if max_z_points.size > 0:
        ax.scatter(max_z_points[:, 1], max_z_points[:, 2], color='red')

        # Fit parametric 3D spline: [X(t), Y(t), Z(t)]
        tck, u = splprep([max_z_points[:, 0], max_z_points[:, 1], max_z_points[:, 2]], s=0)
        from scipy.optimize import root_scalar

        # Find parameter values u_exact corresponding to exact y-slice locations
        y_targets = max_z_points[:, 1]
        u_exact = []
        for y_target in y_targets:
            res = root_scalar(lambda u_: splev(u_, tck)[1] - y_target, bracket=[0, 1], method='brentq')
            u_exact.append(res.root)
        u_exact = np.array(u_exact)

        x_eval, y_eval, z_eval = splev(u_exact, tck)
        dx, dy, dz = splev(u_exact, tck, der=1)

        # Tangent vectors in 3D
        tangents = np.vstack([dx, dy, dz]).T
        tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]

        # Approximate normal vectors (2D projection into Y-Z)
        tangent_yz = tangents[:, 1:3]
        normals_yz = np.vstack([-tangent_yz[:, 1], tangent_yz[:, 0]]).T
        normals_yz /= np.linalg.norm(normals_yz, axis=1)[:, np.newaxis]

        # Save 3D tangent vectors at each slice point
        output_df = pd.DataFrame({
            #'X': x_eval,
            'Y': y_eval,
            #'Z': z_eval,
            'Tangent_X': tangents[:, 0],
            'Tangent_Y': tangents[:, 1],
            'Tangent_Z': tangents[:, 2],
            #'Normal_Y': normals_yz[:, 0],
            #'Normal_Z': normals_yz[:, 1],
        })
        output_df.to_csv("spline_vectors_at_slices.csv", index=False)

        # Plot projection and vectors in Y-Z plane
        ax.plot(y_eval, z_eval, color='black')
        skip = 1
        ax.quiver(
            y_eval[::skip], z_eval[::skip],
            normals_yz[::skip, 0], normals_yz[::skip, 1],
            angles='xy', scale_units='xy', scale=1, color='C2', width=0.005, label='Normal Vectors'
        )
        ax.quiver(
            y_eval[::skip], z_eval[::skip],
            tangent_yz[::skip, 0], tangent_yz[::skip, 1],
            angles='xy', scale_units='xy', scale=1, color='C0', width=0.005, label='Tangent Vectors'
        )

        
        ax.set_xlabel('Y',  fontsize=22, fontname="Times New Roman")
        ax.set_ylabel('Z',  fontsize=22, fontname="Times New Roman")
        ax.set_aspect('equal', adjustable='box')
        #ax.legend()
        ax.grid(False)
        
        F = plt.gcf()
        Size = F.get_size_inches()
        F.set_size_inches(Size[0]*1.5, Size[1]*1.5, forward=True)
    
    
        plt.tight_layout()
        plt.show()

def plot_surface_coords_by_rank(file_paths, zoom_percentile=0.0):
    """
    Reads surface coordinate CSV files from multiple MPI ranks, plots a 3D scatter plot
    with each rank in a different color, and zooms into the central region of the point cloud.

    Parameters:
    - file_paths: List of strings, paths to the CSV files (one per MPI rank)
    - dpi: Resolution of the plot
    - zoom_percentile: Percentile to trim on each end (e.g., 0.25 zooms to 25th–75th percentile)
    """

    plt.style.use('seaborn-v0_8-deep')
    
    
    # Load all CSV files
    dfs = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Define distinct colors and labels
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    labels = [f'Process {i}' for i in range(len(file_paths))]

    # Compute zoom region
    lower = zoom_percentile
    upper = 1 - zoom_percentile
    xlim = (combined_df['X'].quantile(lower), combined_df['X'].quantile(upper))
    ylim = (combined_df['Y'].quantile(lower), combined_df['Y'].quantile(upper))
    zlim = (combined_df['Z'].quantile(lower), combined_df['Z'].quantile(upper))


    # Create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each rank with different color
    for i, df in enumerate(dfs):
        ax.scatter(df['X'], df['Y'], df['Z'], s=8, alpha=0.6, color=colors[i % len(colors)], label=labels[i])

    # Apply zoom
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Turn off axis and background
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)
    
    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
   

    plt.legend(frameon=False, loc='upper left', prop={'size': 16, 'family': 'Times New Roman'}, ncol=3)
    plt.tight_layout()
    plt.savefig('Plots/non_planar_surf_decomp.png')
    #plt.show()


def plot_surface_with_vectors(tangent_normal_csv, file_paths):
    """
    Plots Y-Z projection of surface coordinates with overlaid tangent and normal vectors.

    Parameters:
    - tangent_normal_file: Path to the CSV containing tangent and normal vectors.
    - surface_coords_pattern: Glob pattern to match MPI process coordinate files.
    - figsize: Size of the figure (width, height).
    - dpi: Resolution of the plot.
    - scatter_color: Color of surface points.
    - scatter_alpha: Transparency of surface points.
    - scatter_size: Size of scatter markers.
    - tangent_color: Color of tangent vectors.
    - normal_color: Color of normal vectors.
    - vector_width: Width of the quiver arrows.
    """
    
    plt.style.use('seaborn-v0_8-deep')
     
    # Load tangent and normal data
    df_vec = pd.read_csv(tangent_normal_csv)
    
    print(df_vec.head())
    
    # Load all CSV files
    dfs = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    
    labels = [f'Process {i}' for i in range(len(file_paths))]

    
    
    # Create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot()
    
    # Plot each rank with different color
    for i, df in enumerate(dfs):
        ax.scatter(df['Y'], df['Z'], s=1, alpha=0.2, color=colors[i % len(colors)], label=labels[i])
        
    # Overlay tangent vectors
    ax.quiver(df_vec["Y"], df_vec["Z"], df_vec["Tangent_Y"], df_vec["Tangent_Z"],
              angles='xy', scale_units='xy', scale=1, color="red", width=0.004)

    # Overlay normal vectors
    ax.quiver(df_vec["Y"], df_vec["Z"], df_vec["Normal_Y"], df_vec["Normal_Z"],
              angles='xy', scale_units='xy', scale=1, color="black", width=0.004)


    # Final touches
    ax.axis("equal")
    ax.axis('off')
    
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)
    
    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    plt.tight_layout()
    plt.show()
    
    
def main():
    surface_path = "surface_coordinates.csv"
    slice_path = "slice_locations.csv"
    
    file_paths = [
    "data/surface_coords_1.csv",
    "data/surface_coords_3.csv",
    "data/surface_coords_4.csv",
    "data/surface_coords_5.csv",
    "data/surface_coords_7.csv"
    ]  
    
    tangent_normal_csv = "data/tangent_normals_output.csv"
    
    #plot_surface_coords_by_rank(file_paths)
    
    #surface_df = pd.read_csv("surface_coordinates.csv")
    #plot_surface_with_vectors(tangent_normal_csv, file_paths)

    surface_df, y_slices = load_data(surface_path, slice_path)
    #max_z_points = extract_max_z(surface_df, y_slices)

    plot_3d_surface_with_slices(surface_df, y_slices)
    #plot_yz_projection_with_vectors(max_z_points)
    
    #plot_xy_projection_with_slices(surface_df, y_slices)
    

if __name__ == "__main__":
    main()
