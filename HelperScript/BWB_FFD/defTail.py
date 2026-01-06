#!/usr/bin/env python3
"""
Single SU2-style FFD box from a surface VTU.

User-specified bounds:
  - X in [36, 38] meters
  - Y in [0, 4] meters

Z bounds:
  - estimated automatically from local surface points (robust percentiles)
  - padded by PAD_Z * local_thickness
  - optionally extended further by:
      * absolute meters (Z_EXT_ABS_M), and/or
      * fraction of local thickness (Z_EXT_FRAC)

Also computes and prints the quarter-chord position for the box:
  x_c4 = XMIN + 0.25*(XMAX - XMIN)
"""

import numpy as np
import pyvista as pv
import subprocess

# -----------------------------
# Inputs
# -----------------------------
VTU_FILE = "surface_deformed.vtu"

# Hard bounds (meters)
XMIN = 36.0
XMAX = 38.0
YMIN = 0.0
YMAX = 4.0

# SU2 degrees (same conventions as before)
DEG_X = 1
DEG_Y = 1
DEG_Z = 1

BOX_NAME = "FUSE_END"

# Robust percentiles for automatic Z bounds (avoid stray outliers)
ZLO_PCTL = 1.0
ZHI_PCTL = 99.0

# Padding relative to local z-thickness (same style as your script)
PAD_Z = 0.15

# EXTRA Z extension beyond the padded auto-bounds:
Z_EXT_ABS_M = 0.25   # meters added to BOTH -Z and +Z (set 0.0 to disable)
Z_EXT_FRAC  = 0.10   # fraction of local thickness added to BOTH sides (set 0.0 to disable)

# If too few points exist in the strict [XMIN,XMAX]x[YMIN,YMAX] window,
# expand the window by these factors until enough points are found.
EXPAND_FACTORS = [1.0, 1.5, 2.5, 4.0]
MIN_POINTS_FOR_Z = 50

# -----------------------------
# set_ffd_design_var.py (optional)
# -----------------------------
RUN_SET_FFD = False  # True => actually run; False => dry-run print
SET_FFD_SCRIPT = "set_ffd_design_var.py"
MARKERS = "'wing_top, wing_bottom, tip_top, tip_bottom, TE, tip_TE'"

# Visualization toggles
SHOW_MESH_EDGES = False
MESH_OPACITY = 0.60
DRAW_BOX = True
DRAW_CTRL_POINTS = True
CTRL_POINT_SIZE = 12


def fmt_pt(p):
    return f"{p[0]:.10e}, {p[1]:.10e}, {p[2]:.10e}"


def su2_hexa_corners(xmin0, xmax0, y0, zmin0, zmax0,
                     xmin1, xmax1, y1, zmin1, zmax1):
    """
    Typical SU2 hexa corner ordering:
      1:(xmin,y0,zmin) 2:(xmax,y0,zmin) 3:(xmax,y1,zmin) 4:(xmin,y1,zmin)
      5:(xmin,y0,zmax) 6:(xmax,y0,zmax) 7:(xmax,y1,zmax) 8:(xmin,y1,zmax)
    """
    p1 = np.array([xmin0, y0, zmin0])
    p2 = np.array([xmax0, y0, zmin0])
    p3 = np.array([xmax1, y1, zmin1])
    p4 = np.array([xmin1, y1, zmin1])
    p5 = np.array([xmin0, y0, zmax0])
    p6 = np.array([xmax0, y0, zmax0])
    p7 = np.array([xmax1, y1, zmax1])
    p8 = np.array([xmin1, y1, zmax1])
    return [p1, p2, p3, p4, p5, p6, p7, p8]


def hexa_wireframe(corners):
    p1, p2, p3, p4, p5, p6, p7, p8 = corners
    edges = [
        (p1, p2), (p2, p3), (p3, p4), (p4, p1),
        (p5, p6), (p6, p7), (p7, p8), (p8, p5),
        (p1, p5), (p2, p6), (p3, p7), (p4, p8),
    ]
    pts = []
    lines = []
    idx = 0
    for a, b in edges:
        pts.append(a); pts.append(b)
        lines.extend([2, idx, idx + 1])
        idx += 2
    poly = pv.PolyData(np.array(pts))
    poly.lines = np.array(lines)
    return poly


def trilinear_ctrl_points(corners, deg_x, deg_y, deg_z):
    p1, p2, p3, p4, p5, p6, p7, p8 = corners
    nx, ny, nz = deg_x + 1, deg_y + 1, deg_z + 1

    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)
    zeta = np.linspace(0.0, 1.0, nz)

    ctrl = np.zeros((nx, ny, nz, 3))
    for i, x in enumerate(xi):
        for j, y in enumerate(eta):
            for k, z in enumerate(zeta):
                ctrl[i, j, k, :] = (
                    (1-x)*(1-y)*(1-z)*p1 +
                    x*(1-y)*(1-z)*p2 +
                    x*y*(1-z)*p3 +
                    (1-x)*y*(1-z)*p4 +
                    (1-x)*(1-y)*z*p5 +
                    x*(1-y)*z*p6 +
                    x*y*z*p7 +
                    (1-x)*y*z*p8
                )
    return ctrl


def run_set_ffd_for_box(box_name, ni, nj, nk, run=False):
    cmd = [
        SET_FFD_SCRIPT,
        "-i", str(ni),
        "-j", str(nj),
        "-k", str(nk),
        "-b", box_name,
        "-m", MARKERS,
    ]
    if run:
        subprocess.run(cmd, check=True)
    else:
        print(" ".join(cmd))


def estimate_local_z_bounds(points):
    """
    Estimate z-bounds from points in an (expanded) XY window around [XMIN,XMAX]x[YMIN,YMAX].
    Uses robust percentiles then applies PAD_Z based on local z thickness.
    Returns zlo_raw, zhi_raw (percentile bounds), plus sample info.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    xmid = 0.5 * (XMIN + XMAX)
    ymid = 0.5 * (YMIN + YMAX)
    hx = 0.5 * (XMAX - XMIN)
    hy = 0.5 * (YMAX - YMIN)

    for f in EXPAND_FACTORS:
        xmin = xmid - f * hx
        xmax = xmid + f * hx
        ymin = ymid - f * hy
        ymax = ymid + f * hy

        mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        n = int(mask.sum())
        if n >= MIN_POINTS_FOR_Z:
            zs = z[mask]
            zlo = float(np.percentile(zs, ZLO_PCTL))
            zhi = float(np.percentile(zs, ZHI_PCTL))
            return zlo, zhi, n, (xmin, xmax, ymin, ymax)

    # Fallback: global
    return float(z.min()), float(z.max()), int(points.shape[0]), ("GLOBAL", "GLOBAL", "GLOBAL", "GLOBAL")


def main():
    mesh = pv.read(VTU_FILE)
    pts = mesh.points

    # -----------------------------
    # Quarter-chord (from box X bounds)
    # -----------------------------
    chord_box = XMAX - XMIN
    x_c4 = XMIN + 0.25 * chord_box
    y_mid = 0.5 * (YMIN + YMAX)

    # -----------------------------
    # Automatic Z bounds from local points
    # -----------------------------
    zlo_raw, zhi_raw, n_used, win = estimate_local_z_bounds(pts)
    zth = max(zhi_raw - zlo_raw, 1e-12)

    # First apply your "padding" convention
    zmin = zlo_raw - PAD_Z * zth
    zmax = zhi_raw + PAD_Z * zth

    # Then apply extra user-requested symmetric extension
    z_extra = float(Z_EXT_ABS_M) + float(Z_EXT_FRAC) * zth
    zmin -= z_extra
    zmax += z_extra

    # You might also want a "c/4 point" in 3D; choose z-mid of final bounds:
    z_mid = 0.5 * (zmin + zmax)

    # -----------------------------
    # Corners (no taper; same X/Z at both Y stations)
    # -----------------------------
    corners = su2_hexa_corners(
        XMIN, XMAX, YMIN, zmin, zmax,
        XMIN, XMAX, YMAX, zmin, zmax
    )

    # -----------------------------
    # PRINT SU2 block + c/4 info
    # -----------------------------
    print("\n# --------------------------------------------")
    print("# SU2 FFD: single box from VTU (fuselage end region)")
    print("# Axes: chord=X, span=Y, dihedral/thickness=Z")
    print("# --------------------------------------------")
    print(f"# VTU_FILE   = {VTU_FILE}")
    print(f"# X bounds   = [{XMIN:.6f}, {XMAX:.6f}]")
    print(f"# Y bounds   = [{YMIN:.6f}, {YMAX:.6f}]")
    print(f"# Z pctl     = ({ZLO_PCTL:.1f}, {ZHI_PCTL:.1f})")
    print(f"# PAD_Z      = {PAD_Z:.3f}  (fraction of local thickness)")
    print(f"# Z_EXT_ABS  = {Z_EXT_ABS_M:.6f} m (each side)")
    print(f"# Z_EXT_FRAC = {Z_EXT_FRAC:.3f}  (each side, fraction of thickness)")
    print(f"# Z sample   = n={n_used}, XY window={win}")
    print(f"# Local zth  = {zth:.6f}")
    print(f"# Final Z    = [{zmin:.6f}, {zmax:.6f}]")
    print("# --------------------------------------------\n")

    print("# Quarter-chord info (from box X bounds):")
    print(f"#   chord_box = {chord_box:.6f}")
    print(f"#   x_c/4     = {x_c4:.6f}")
    print(f"#   y_mid     = {y_mid:.6f}")
    print(f"#   z_mid     = {z_mid:.6f}")
    print(f"#   c/4 point = ({x_c4:.6f}, {y_mid:.6f}, {z_mid:.6f})\n")

    print(f"# Box: {BOX_NAME}")
    print(f"FFD_DEGREE = ({int(DEG_X)}, {int(DEG_Y)}, {int(DEG_Z)})")
    print("FFD_DEFINITION = \\")
    print(
        "  "
        f"({BOX_NAME}, "
        f"{fmt_pt(corners[0])}, {fmt_pt(corners[1])}, "
        f"{fmt_pt(corners[2])}, {fmt_pt(corners[3])}, "
        f"{fmt_pt(corners[4])}, {fmt_pt(corners[5])}, "
        f"{fmt_pt(corners[6])}, {fmt_pt(corners[7])})\n"
    )

    # -----------------------------
    # set_ffd_design_var.py call (kept consistent with your earlier script)
    # -----------------------------
    print("\n# --------------------------------------------")
    print("# set_ffd_design_var.py call (single box)")
    print("# --------------------------------------------\n")
    run_set_ffd_for_box(BOX_NAME, int(DEG_X), int(DEG_Y), int(DEG_Z), run=RUN_SET_FFD)

    # -----------------------------
    # DRAW (PyVista)
    # -----------------------------
    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=MESH_OPACITY, show_edges=SHOW_MESH_EDGES)

    if DRAW_BOX:
        wf = hexa_wireframe(corners)
        pl.add_mesh(wf, color="black", line_width=3)

    if DRAW_CTRL_POINTS:
        ctrl = trilinear_ctrl_points(corners, int(DEG_X), int(DEG_Y), int(DEG_Z))
        ctrl_pts = ctrl.reshape(-1, 3)
        pl.add_points(
            ctrl_pts,
            render_points_as_spheres=True,
            point_size=CTRL_POINT_SIZE,
            color="red",
        )

    # Optional: show the c/4 point
    pl.add_points(
        np.array([[x_c4, y_mid, z_mid]]),
        render_points_as_spheres=True,
        point_size=18,
        color="blue",
    )

    pl.add_axes()
    pl.show_grid()
    pl.show()


if __name__ == "__main__":
    main()
