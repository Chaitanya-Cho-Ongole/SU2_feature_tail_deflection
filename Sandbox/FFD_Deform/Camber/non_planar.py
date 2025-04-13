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
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(surface_df['X'], surface_df['Y'], surface_df['Z'], c='gray', alpha=0.1, s=0.5)

    x_min, x_max = surface_df['X'].min(), surface_df['X'].max()
    z_min, z_max = surface_df['Z'].min(), surface_df['Z'].max()

    for y in y_slices:
        X_plane, Z_plane = np.meshgrid([x_min, x_max], [z_min, z_max])
        Y_plane = np.full_like(X_plane, y)
        ax.plot_surface(X_plane, Y_plane, Z_plane, color='blue', alpha=0.2, edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Mesh with Slice Planes')
    plt.tight_layout()
    plt.show()

def plot_yz_projection_with_vectors(max_z_points):
    fig, ax = plt.subplots(dpi=300)
    if max_z_points.size > 0:
        ax.scatter(max_z_points[:, 0], max_z_points[:, 1], color='red')

        # Fit spline and compute derivatives
        spline = UnivariateSpline(max_z_points[:, 0], max_z_points[:, 1], s=0)
        y_dense = np.linspace(max_z_points[:, 0].min(), max_z_points[:, 0].max(), 300)
        z_dense = spline(y_dense)
        dz_dy = spline.derivative()(y_dense)

        # Compute normalized tangent and normal vectors
        tangent_vectors = np.vstack([np.ones_like(dz_dy), dz_dy]).T
        normal_vectors = np.vstack([-dz_dy, np.ones_like(dz_dy)]).T
        tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]
        normal_vectors /= np.linalg.norm(normal_vectors, axis=1)[:, np.newaxis]

        skip = 10
        ax.plot(y_dense, z_dense, color='black', label='Spline')
        ax.quiver(
            y_dense[::skip], z_dense[::skip],
            normal_vectors[::skip, 0], normal_vectors[::skip, 1],
            angles='xy', scale_units='xy', scale=3, color='blue', width=0.007, label='Normal Vectors'
        )
        ax.quiver(
            y_dense[::skip], z_dense[::skip],
            tangent_vectors[::skip, 0], tangent_vectors[::skip, 1],
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
