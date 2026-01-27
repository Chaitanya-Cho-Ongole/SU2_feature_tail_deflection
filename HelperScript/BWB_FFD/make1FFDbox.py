#!/usr/bin/env python3
"""
ONE SU2-style FFD box from a deformed surface VTU (single box encapsulates entire geometry).

Changes vs your TWO-box script:
  - Removed fuselage/wing splitting and all explicit Y-range logic.
  - Compute one GLOBAL spanwise profile over the whole geometry.
  - Apply ONE set of padding (LE/TE/Z) globally (constant).
  - Build ONE tapered hexa box using the end stations (y_min and y_max).
  - Print exactly ONE FFD_DEGREE / FFD_DEFINITION block.
  - Print exactly ONE set_ffd_design_var.py command.
  - Visualization draws only this one box + its control points.
"""

import numpy as np
import pyvista as pv
import subprocess

# -----------------------------
# Inputs
# -----------------------------
VTU_FILE = "surface_deformed.vtu"

# Single box name
BOX_PREFIX = "FULL"
BOX_NAME = f"{BOX_PREFIX}_000"

# Degrees (explicit)
DEG_Z = 1
DEG_Y = 12
DEG_X = 16

# -----------------------------
# Global padding (fractions)
# -----------------------------
PAD_LE = 0.15
PAD_TE = 0.20
PAD_Z  = 0.50

# -----------------------------
# Robust percentiles (global)
# -----------------------------
LE_PCTL  = 0.5
TE_PCTL  = 99.5
ZLO_PCTL = 0.5
ZHI_PCTL = 99.5

# Profiling resolution and neighborhood
NY_STATIONS = 256
WINDOW_FRAC = 2.5

# -----------------------------
# set_ffd_design_var.py
# -----------------------------
RUN_SET_FFD = False  # True => actually run; False => dry-run print
SET_FFD_SCRIPT = "set_ffd_design_var.py"

# NOTE: You currently include quotes inside MARKERS; that will be passed literally.
# If set_ffd_design_var.py expects the quotes, keep it. Otherwise remove the outer quotes.
MARKERS = "'wing_top, wing_bottom, tip_top, tip_bottom, TE, tip_TE' "

# Visualization toggles
SHOW_MESH_EDGES = False
MESH_OPACITY = 0.60
DRAW_BOXES = True
DRAW_CTRL_POINTS = True
CTRL_POINT_SIZE = 10


def spanwise_profiles(points, y_stations, half_window,
                      le_pctl, te_pctl, zlo_pctl, zhi_pctl):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    xLE = np.empty_like(y_stations)
    xTE = np.empty_like(y_stations)
    zlo = np.empty_like(y_stations)
    zhi = np.empty_like(y_stations)

    for j, yj in enumerate(y_stations):
        mask = np.abs(y - yj) <= half_window

        if mask.sum() < 50:
            mask = np.abs(y - yj) <= 2.5 * half_window
        if mask.sum() < 10:
            mask = np.abs(y - yj) <= 5.0 * half_window
        if mask.sum() < 10:
            idx = np.argsort(np.abs(y - yj))[:200]
            mask = np.zeros_like(y, dtype=bool)
            mask[idx] = True

        xs = x[mask]
        zs = z[mask]

        xLE[j] = np.percentile(xs, le_pctl)
        xTE[j] = np.percentile(xs, te_pctl)
        zlo[j] = np.percentile(zs, zlo_pctl)
        zhi[j] = np.percentile(zs, zhi_pctl)

    xTE = np.maximum(xTE, xLE + 1e-12)
    return xLE, xTE, zlo, zhi


def fmt_pt(p):
    return f"{p[0]:.10e}, {p[1]:.10e}, {p[2]:.10e}"


def su2_hexa_corners(xmin0, xmax0, y0, zmin0, zmax0,
                     xmin1, xmax1, y1, zmin1, zmax1):
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


def apply_padding_const(xLE, xTE, zlo, zhi, pad_le, pad_te, pad_z):
    chord = np.maximum(xTE - xLE, 1e-12)
    zth   = np.maximum(zhi - zlo, 1e-12)

    xmin = xLE - pad_le * chord
    xmax = xLE + (1.0 + pad_te) * chord
    zmin = zlo - pad_z * zth
    zmax = zhi + pad_z * zth
    return xmin, xmax, zmin, zmax


def main():
    mesh = pv.read(VTU_FILE)
    pts = mesh.points

    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    # Stations: dense plus forced endpoints
    y_dense = np.linspace(y_min, y_max, NY_STATIONS)
    y_st = np.unique(np.sort(np.r_[y_dense, [y_min, y_max]]))

    span = y_max - y_min
    dy_nom = span / max(NY_STATIONS - 1, 1)
    half_window = 0.5 * WINDOW_FRAC * dy_nom

    # GLOBAL profiles (one set)
    xLE, xTE, zlo, zhi = spanwise_profiles(
        pts, y_st, half_window,
        LE_PCTL, TE_PCTL, ZLO_PCTL, ZHI_PCTL
    )

    # Apply GLOBAL padding (constant)
    xmin, xmax, zmin, zmax = apply_padding_const(
        xLE, xTE, zlo, zhi, PAD_LE, PAD_TE, PAD_Z
    )

    # Use the end stations to build a tapered hexa (like your original logic)
    i0 = int(np.argmin(np.abs(y_st - y_min)))
    i1 = int(np.argmin(np.abs(y_st - y_max)))

    corners = su2_hexa_corners(
        xmin[i0], xmax[i0], y_st[i0], zmin[i0], zmax[i0],
        xmin[i1], xmax[i1], y_st[i1], zmin[i1], zmax[i1],
    )

    # -----------------------------
    # PRINT SU2 blocks (single box)
    # -----------------------------
    print("\n# --------------------------------------------")
    print("# SU2 FFD: ONE tapered box from VTU (global bounds over entire geometry)")
    print("# Axes: chord=X, span=Y, thickness=Z")
    print(f"# Mesh span: y_min={y_min:.10e}, y_max={y_max:.10e}")
    print("# Degrees:")
    print(f"#  FULL: deg = ({DEG_X}, {DEG_Y}, {DEG_Z})")
    print("# Padding (fractions):")
    print(f"#  PAD_LE={PAD_LE}, PAD_TE={PAD_TE}, PAD_Z={PAD_Z}")
    print("# Robust percentiles:")
    print(f"#  LE={LE_PCTL}, TE={TE_PCTL}, ZLO={ZLO_PCTL}, ZHI={ZHI_PCTL}")
    print("# --------------------------------------------\n")

    print(f"FFD_DEGREE = ({DEG_X}, {DEG_Y}, {DEG_Z})")
    print("FFD_DEFINITION = \\")
    print(
        f"  ({BOX_NAME}, "
        f"{fmt_pt(corners[0])}, {fmt_pt(corners[1])}, {fmt_pt(corners[2])}, {fmt_pt(corners[3])}, "
        f"{fmt_pt(corners[4])}, {fmt_pt(corners[5])}, {fmt_pt(corners[6])}, {fmt_pt(corners[7])})"
    )
    print("")

    # -----------------------------
    # set_ffd_design_var.py call (single box)
    # -----------------------------
    print("\n# --------------------------------------------")
    print("# set_ffd_design_var.py call (ONE total)")
    print("# --------------------------------------------\n")

    # Keep your convention: pass degrees (not deg+1).
    run_set_ffd_for_box(BOX_NAME, DEG_X, DEG_Y, DEG_Z, run=RUN_SET_FFD)

    # -----------------------------
    # DRAW
    # -----------------------------
    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=MESH_OPACITY, show_edges=SHOW_MESH_EDGES)

    if DRAW_BOXES:
        pl.add_mesh(hexa_wireframe(corners), color="black", line_width=2)

        if DRAW_CTRL_POINTS:
            ctrl = trilinear_ctrl_points(corners, DEG_X, DEG_Y, DEG_Z)
            ctrl_pts = ctrl.reshape(-1, 3)
            pl.add_points(
                ctrl_pts,
                render_points_as_spheres=True,
                point_size=CTRL_POINT_SIZE,
                color="red",
            )

    pl.add_axes()
    pl.show_grid()
    pl.show()


if __name__ == "__main__":
    main()
