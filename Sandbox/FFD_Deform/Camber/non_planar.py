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
    
    ax.scatter(surface_df['X'], surface_df['Y'], surface_df['Z'], c='gray', alpha=0.05, s=0.5)

    x_min, x_max = surface_df['X'].min(), surface_df['X'].max()
    z_min, z_max = surface_df['Z'].min(), surface_df['Z'].max()

    for y in y_slices:
        X_plane, Z_plane = np.meshgrid([x_min, x_max], [z_min, z_max])
        Y_plane = np.full_like(X_plane, y)
        ax.plot_surface(X_plane, Y_plane, Z_plane, color='blue', alpha=0.05, edgecolor='none')

    ax.set_xlabel('X', fontsize=22, fontname="Times New Roman")
    ax.set_ylabel('Y', fontsize=22, fontname="Times New Roman")
    ax.set_zlabel('Z', fontsize=22, fontname="Times New Roman")
    ax.grid(False)
    ax.set_axis_off()
    
    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0]*1.5, Size[1]*1.5, forward=True)
    
    
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

def main():
    surface_path = "surface_coordinates.csv"
    slice_path = "slice_locations.csv"

    surface_df, y_slices = load_data(surface_path, slice_path)
    max_z_points = extract_max_z(surface_df, y_slices)

    #plot_3d_surface_with_slices(surface_df, y_slices)
    plot_yz_projection_with_vectors(max_z_points)

if __name__ == "__main__":
    main()
