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
            max_z_points.append((max_point['Y'], max_point['Z']))
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
    fig, ax = plt.subplots(dpi=300)
    if max_z_points.size > 0:
        ax.scatter(max_z_points[:, 0], max_z_points[:, 1], color='red')

        # Fit spline and compute derivatives at slice locations only
        spline = UnivariateSpline(max_z_points[:, 0], max_z_points[:, 1], s=0)
        y_eval = max_z_points[:, 0]
        z_eval = spline(y_eval)
        dz_dy = spline.derivative()(y_eval)

        # Compute normalized tangent and normal vectors
        tangent_vectors = np.vstack([np.ones_like(dz_dy), dz_dy]).T
        normal_vectors = np.vstack([-dz_dy, np.ones_like(dz_dy)]).T
        tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]
        normal_vectors /= np.linalg.norm(normal_vectors, axis=1)[:, np.newaxis]

        # Save vectors to CSV for slice locations only
        output_df = pd.DataFrame({
            'Y': y_eval,
            'Z': z_eval,
            'Tangent_Y': tangent_vectors[:, 0],
            'Tangent_Z': tangent_vectors[:, 1],
            'Normal_Y': normal_vectors[:, 0],
            'Normal_Z': normal_vectors[:, 1],
        })
        output_df.to_csv("spline_vectors_at_slices.csv", index=False)

        # Plot quivers
        ax.plot(y_eval, z_eval, color='black', label='Spline at Slice Points')
        ax.quiver(
            y_eval, z_eval,
            normal_vectors[:, 0], normal_vectors[:, 1],
            angles='xy', scale_units='xy', scale=3, color='blue', width=0.007, label='Normal Vectors'
        )
        ax.quiver(
            y_eval, z_eval,
            tangent_vectors[:, 0], tangent_vectors[:, 1],
            angles='xy', scale_units='xy', scale=3, color='green', width=0.007, label='Tangent Vectors'
        )

        ax.set_title('Y-Z Projection with Tangent and Normal Vectors')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    surface_path = "surface_coordinates.csv"
    slice_path = "slice_locations.csv"

    surface_df, y_slices = load_data(surface_path, slice_path)
    max_z_points = extract_max_z(surface_df, y_slices)

    plot_3d_surface_with_slices(surface_df, y_slices)
    plot_yz_projection_with_vectors(max_z_points)

if __name__ == "__main__":
    main()
