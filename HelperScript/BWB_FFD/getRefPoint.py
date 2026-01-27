#!/usr/bin/env python3
"""
Highlight the pitching-moment reference point for an aircraft surface VTU using a MAC-based definition.

Axes / sign convention (as you stated):
  - x: chordwise
  - y: spanwise (pitch axis is +Y)
  - z: vertical
  - Fz is positive UP

Key consequence for pitching moment about +Y:
  My depends on x_ref and z_ref, but NOT on y_ref.

Moment-shift relation (about +Y):
  My(ref1) = My(ref0) + Δz*Fx - Δx*Fz
  (no Δy term)

Given MAC info:
  - MAC length: 5.33 m
  - MAC leading edge at: X = 23.857 m, Y = 13.097 m
  - Reference point taken at quarter-chord of MAC:
      X_ref = X_LE + 0.25*MAC
      Y_ref = Y_LE   (this is only the spanwise station; it does not affect My itself)
  - Z_ref is estimated from nearby mesh points at (X_ref, Y_ref) by taking median Z.

This script:
  1) Reads surface_deformed.vtu
  2) Computes (X_ref, Y_ref) from MAC
  3) Estimates Z_ref from mesh
  4) Plots the mesh + a highlighted reference point
"""

import numpy as np
import pyvista as pv

# -----------------------------
# Inputs
# -----------------------------
VTU_FILE = "surface_flow.vtu"

MAC = 5.33
X_LE_MAC = 23.857
Y_LE_MAC = 13.097

FRACTION = 0.55

# Quarter-chord reference in X, same spanwise station in Y
X_REF = X_LE_MAC + FRACTION * MAC
Y_REF = Y_LE_MAC

# Z estimation settings (in meters)
R0 = 0.05        # initial XY radius
R_GROW = 2.0     # expand factor each attempt
MIN_NEAR_PTS = 50
MAX_TRIES = 6

# Visualization
MESH_OPACITY = 0.65
SHOW_MESH_EDGES = False

DRAW_LOCAL_CLOUD = True
LOCAL_CLOUD_SIZE = 6

REF_POINT_SIZE = 18
REF_POINT_COLOR = "red"


def estimate_z_at_xy(points: np.ndarray, x0: float, y0: float) -> tuple[float, np.ndarray]:
    """
    Estimate Z at (x0,y0) by collecting nearby points in XY and taking median Z.
    Returns:
      z_est, near_points (Nx3) used for the estimate (may be empty if fallback used)
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = R0
    near = np.empty((0, 3), dtype=float)

    for _ in range(MAX_TRIES):
        mask = (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2
        if np.count_nonzero(mask) >= MIN_NEAR_PTS:
            near = points[mask]
            return float(np.median(near[:, 2])), near
        r *= R_GROW

    # Fallback: global mid-Z
    z_est = float(0.5 * (z.min() + z.max()))
    return z_est, near


def main():
    mesh = pv.read(VTU_FILE)
    pts = mesh.points

    x_ref = float(X_REF)
    y_ref = float(Y_REF)

    z_ref, near_pts = estimate_z_at_xy(pts, x_ref, y_ref)
    ref_pt = np.array([x_ref, y_ref, z_ref], dtype=float)

    # -----------------------------
    # Print reference point + the "My depends on x,z not y" statement
    # -----------------------------
    print("\n# --------------------------------------------")
    print("# Pitching-moment reference point (about +Y)")
    print("# Axes: x=chordwise, y=spanwise (pitch axis), z=vertical")
    print("# Sign: Fz positive UP")
    print("# Moment shift about +Y:  My1 = My0 + Δz*Fx - Δx*Fz   (no Δy term)")
    print("# --------------------------------------------")

    print(f"MAC               = {MAC:.6f} m")
    print(f"MAC LE (X_LE,Y_LE) = ({X_LE_MAC:.6f}, {Y_LE_MAC:.6f}) m")
    print(f"Quarter-chord X_ref= X_LE + 0.25*MAC = {x_ref:.6f} m")
    print(f"Chosen Y_ref       = Y_LE_MAC = {y_ref:.6f} m  (note: y_ref does NOT affect My)")

    if near_pts.shape[0] > 0:
        print(f"Estimated Z_ref    = median(Z) of {near_pts.shape[0]} nearby points = {z_ref:.6f} m")
    else:
        print(f"Estimated Z_ref    = fallback global mid-Z = {z_ref:.6f} m")

    print(f"\nREFERENCE_POINT = ({ref_pt[0]:.10e}, {ref_pt[1]:.10e}, {ref_pt[2]:.10e})\n")

    # -----------------------------
    # Visualization
    # -----------------------------
    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=MESH_OPACITY, show_edges=SHOW_MESH_EDGES)

    # Show the neighborhood used for Z_ref (optional)
    if DRAW_LOCAL_CLOUD and near_pts.shape[0] > 0:
        local_poly = pv.PolyData(near_pts)
        pl.add_points(
            local_poly,
            render_points_as_spheres=True,
            point_size=LOCAL_CLOUD_SIZE,
        )

    # Highlight reference point
    ref_poly = pv.PolyData(ref_pt.reshape(1, 3))
    pl.add_points(
        ref_poly,
        render_points_as_spheres=True,
        point_size=REF_POINT_SIZE,
        color=REF_POINT_COLOR,
    )

    pl.add_point_labels(
        ref_poly,
        [f"Ref (quarter-chord of MAC)\nX={x_ref:.3f}, Y={y_ref:.3f}, Z={z_ref:.3f}"],
        font_size=14,
        shape=None,
    )

    pl.add_axes()
    pl.show_grid()
    pl.show()


if __name__ == "__main__":
    main()
