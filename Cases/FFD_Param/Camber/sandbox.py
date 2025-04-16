import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    df_candidates = pd.read_csv("surface_coordinates.csv")
    df_normals = pd.read_csv("slice_locations.csv")
    df_spine = pd.read_csv("spine_points.csv")

    # Clean trailing spaces from column names
    df_spine.columns = [col.strip() for col in df_spine.columns]

    return df_candidates, df_normals, df_spine

def compute_centroids(df_candidates, df_normals, y_tol=0.25):
    centroids = []
    for _, row in df_normals.iterrows():
        y_val = row['Y']
        slice_df = df_candidates[
            (df_candidates['Y'] >= y_val - y_tol) & (df_candidates['Y'] <= y_val + y_tol)
        ]
        if not slice_df.empty:
            centroid = slice_df[['X', 'Y', 'Z']].mean().values
            centroids.append((centroid, [row['Xn'], row['Yn'], row['Zn']]))
    return centroids

def plot_3d(ax, df_candidates, centroids_with_normals):
    ax.scatter(df_candidates['X'], df_candidates['Y'], df_candidates['Z'],
               s=0.01, alpha=0.1, color='gray')
    for centroid, normal in centroids_with_normals:
        ax.quiver(*centroid, *normal, length=1.0, color='red')
    ax.set_title('3D View')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_yz_projection(ax, df_candidates, centroids_with_normals):
    ax.scatter(df_candidates['Y'], df_candidates['Z'],
               s=0.01, alpha=0.1, color='gray')
    for centroid, normal in centroids_with_normals:
        y, z = centroid[1], centroid[2]
        ny, nz = normal[1], normal[2]
        ax.quiver(y, z, ny, nz, angles='xy', scale_units='xy', scale=1.0, color='red')
    ax.set_title('YZ Projection')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.axis('equal')

def plot_xy(ax, df_candidates, df_spine):
    ax.scatter(df_candidates['X'], df_candidates['Y'],
               s=0.01, alpha=0.1, color='gray', label='Candidate Points')
    ax.plot(df_spine['X'], df_spine['Y'], color='blue', linewidth=1.5, label='Spine')
    ax.set_title('XY Plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

def plot_xz(ax, df_candidates, df_spine):
    ax.scatter(df_candidates['X'], df_candidates['Z'],
               s=0.01, alpha=0.1, color='gray', label='Candidate Points')
    ax.plot(df_spine['X'], df_spine['Z'], color='green', linewidth=1.5, label='Spine')
    ax.set_title('XZ Plane')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.legend()

if __name__ == "__main__":
    df_candidates, df_normals, df_spine = load_data()
    centroids_with_normals = compute_centroids(df_candidates, df_normals)

    fig = plt.figure(figsize=(14, 10))
    
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d(ax1, df_candidates, centroids_with_normals)

    ax2 = fig.add_subplot(222)
    plot_yz_projection(ax2, df_candidates, centroids_with_normals)

    ax3 = fig.add_subplot(223)
    plot_xy(ax3, df_candidates, df_spine)

    ax4 = fig.add_subplot(224)
    plot_xz(ax4, df_candidates, df_spine)

    plt.tight_layout()
    plt.show()
