#!/usr/bin/env python3
"""
TWO SU2-style FFD boxes from a deformed surface VTU, with explicit GAP/OVERLAP in Y:

  - FUSELAGE box spans: [Y_FUSE_START, Y_FUSE_END]
  - WING box spans:     [Y_WING_START, Y_WING_END]

Update in this version:
  - Wing padding is defined separately at the wing ROOT and TIP for:
      * LE padding
      * TE padding
      * Z padding   <-- NEW (root vs tip)
    and we linearly interpolate padding along Y between root and tip.

Everything else stays the same.
"""

import numpy as np
import pyvista as pv
import subprocess

# -----------------------------
# Inputs
# -----------------------------
VTU_FILE = "surface_deformed.vtu"

# Explicit Y ranges (mesh Y units)
Y_FUSE_START = None   # None => use y_min
Y_FUSE_END   = 12.0
Y_WING_START = 11.0   # can be >, =, or < Y_FUSE_END
Y_WING_END   = None   # None => use y_max

# Degrees (explicit)
DEG_Z = 1
DEG_Y_FUSE = 4
DEG_Y_WING = 8

DEG_X_FUSE = 12
DEG_X_WING = 8

# -----------------------------
# Fuselage padding (fractions)
# -----------------------------
PAD_LE_FUSE = 0.10
PAD_TE_FUSE = 0.15
PAD_Z_FUSE  = 0.20

# -----------------------------
# Wing padding (fractions) - ROOT vs TIP for LE/TE/Z
#   - Root is at Y_WING_START
#   - Tip  is at Y_WING_END (or y_max if None)
#   - Padding is linearly interpolated across the wing Y-range
# -----------------------------
PAD_LE_WING_ROOT = 0.08
PAD_TE_WING_ROOT = 0.08
PAD_Z_WING_ROOT  = 0.35   

PAD_LE_WING_TIP  = 0.40
PAD_TE_WING_TIP  = 0.30
PAD_Z_WING_TIP   = 1.60   

# -----------------------------
# Region-specific robust percentiles
# -----------------------------
LE_PCTL_FUSE  = 0.1
TE_PCTL_FUSE  = 99.9
ZLO_PCTL_FUSE = 0.1
ZHI_PCTL_FUSE = 99.9

LE_PCTL_WING  = 1.0
TE_PCTL_WING  = 99.0
ZLO_PCTL_WING = 1.0
ZHI_PCTL_WING = 99.0

# Profiling resolution and neighborhood
NY_STATIONS = 256
WINDOW_FRAC = 2.5

BOX_PREFIX = "WING"

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


def snap_station(y_st, y_query):
    idx = int(np.argmin(np.abs(y_st - y_query)))
    return idx, float(y_st[idx])


def apply_padding_const(xLE, xTE, zlo, zhi, pad_le, pad_te, pad_z):
    chord = np.maximum(xTE - xLE, 1e-12)
    zth   = np.maximum(zhi - zlo, 1e-12)

    xmin = xLE - pad_le * chord
    xmax = xLE + (1.0 + pad_te) * chord
    zmin = zlo - pad_z * zth
    zmax = zhi + pad_z * zth
    return xmin, xmax, zmin, zmax


def apply_padding_wing_root_tip(xLE, xTE, zlo, zhi, y_st,
                                y_root, y_tip,
                                pad_le_root, pad_te_root, pad_z_root,
                                pad_le_tip,  pad_te_tip,  pad_z_tip):
    """
    Wing padding with separate LE/TE/Z padding at root and tip, linearly interpolated in Y.

      pad_le(y) = lerp(pad_le_root -> pad_le_tip)
      pad_te(y) = lerp(pad_te_root -> pad_te_tip)
      pad_z(y)  = lerp(pad_z_root  -> pad_z_tip)
    """
    chord = np.maximum(xTE - xLE, 1e-12)
    zth   = np.maximum(zhi - zlo, 1e-12)

    denom = max(y_tip - y_root, 1e-12)
    t = (y_st - y_root) / denom
    t = np.clip(t, 0.0, 1.0)

    pad_le = (1.0 - t) * pad_le_root + t * pad_le_tip
    pad_te = (1.0 - t) * pad_te_root + t * pad_te_tip
    pad_z  = (1.0 - t) * pad_z_root  + t * pad_z_tip

    xmin = xLE - pad_le * chord
    xmax = xLE + (1.0 + pad_te) * chord
    zmin = zlo - pad_z * zth
    zmax = zhi + pad_z * zth
    return xmin, xmax, zmin, zmax


def main():
    mesh = pv.read(VTU_FILE)
    pts = mesh.points

    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    # Fill None defaults
    y_f0 = y_min if Y_FUSE_START is None else float(Y_FUSE_START)
    y_f1 = float(Y_FUSE_END)
    y_w0 = float(Y_WING_START)
    y_w1 = y_max if Y_WING_END is None else float(Y_WING_END)

    # Basic validity checks (each box must have positive span)
    if not (y_min <= y_f0 < y_f1 <= y_max):
        raise ValueError(
            f"FUSE box must satisfy y_min <= Y_FUSE_START < Y_FUSE_END <= y_max.\n"
            f"Got y_min={y_min:.6e}, Y_FUSE_START={y_f0:.6e}, Y_FUSE_END={y_f1:.6e}, y_max={y_max:.6e}"
        )
    if not (y_min <= y_w0 < y_w1 <= y_max):
        raise ValueError(
            f"WING box must satisfy y_min <= Y_WING_START < Y_WING_END <= y_max.\n"
            f"Got y_min={y_min:.6e}, Y_WING_START={y_w0:.6e}, Y_WING_END={y_w1:.6e}, y_max={y_max:.6e}"
        )

    # Profiling stations (dense) + force in requested endpoints
    y_dense = np.linspace(y_min, y_max, NY_STATIONS)
    y_st = np.unique(np.sort(np.r_[y_dense, [y_f0, y_f1, y_w0, y_w1]]))

    span = y_max - y_min
    dy_nom = span / max(NY_STATIONS - 1, 1)
    half_window = 0.5 * WINDOW_FRAC * dy_nom

    # Profiles
    xLE_f, xTE_f, zlo_f, zhi_f = spanwise_profiles(
        pts, y_st, half_window,
        LE_PCTL_FUSE, TE_PCTL_FUSE, ZLO_PCTL_FUSE, ZHI_PCTL_FUSE
    )
    xLE_w, xTE_w, zlo_w, zhi_w = spanwise_profiles(
        pts, y_st, half_window,
        LE_PCTL_WING, TE_PCTL_WING, ZLO_PCTL_WING, ZHI_PCTL_WING
    )

    # Apply padding
    xmin_f, xmax_f, zmin_f, zmax_f = apply_padding_const(
        xLE_f, xTE_f, zlo_f, zhi_f, PAD_LE_FUSE, PAD_TE_FUSE, PAD_Z_FUSE
    )

    xmin_w, xmax_w, zmin_w, zmax_w = apply_padding_wing_root_tip(
        xLE_w, xTE_w, zlo_w, zhi_w, y_st,
        y_root=y_w0, y_tip=y_w1,
        pad_le_root=PAD_LE_WING_ROOT, pad_te_root=PAD_TE_WING_ROOT, pad_z_root=PAD_Z_WING_ROOT,
        pad_le_tip=PAD_LE_WING_TIP,   pad_te_tip=PAD_TE_WING_TIP,   pad_z_tip=PAD_Z_WING_TIP
    )

    # Snap requested Y's to stations
    i_f0, y_f0s = snap_station(y_st, y_f0)
    i_f1, y_f1s = snap_station(y_st, y_f1)
    i_w0, y_w0s = snap_station(y_st, y_w0)
    i_w1, y_w1s = snap_station(y_st, y_w1)

    # Explicit degrees in X
    degx_fuse = int(DEG_X_FUSE)
    degx_wing = int(DEG_X_WING)

    # Build corners (tapered using endpoint stations)
    corners_fuse = su2_hexa_corners(
        xmin_f[i_f0], xmax_f[i_f0], y_st[i_f0], zmin_f[i_f0], zmax_f[i_f0],
        xmin_f[i_f1], xmax_f[i_f1], y_st[i_f1], zmin_f[i_f1], zmax_f[i_f1]
    )
    corners_wing = su2_hexa_corners(
        xmin_w[i_w0], xmax_w[i_w0], y_st[i_w0], zmin_w[i_w0], zmax_w[i_w0],
        xmin_w[i_w1], xmax_w[i_w1], y_st[i_w1], zmin_w[i_w1], zmax_w[i_w1]
    )

    box_fuse = f"{BOX_PREFIX}_FUSE"
    box_wing = f"{BOX_PREFIX}_WING"

    # -----------------------------
    # PRINT SU2 blocks
    # -----------------------------
    print("\n# --------------------------------------------")
    print("# SU2 FFD: TWO tapered boxes from VTU (explicit Y ranges; gap/overlap allowed)")
    print("# Axes: chord=X, span=Y, thickness=Z")
    print(f"# Mesh span: y_min={y_min:.10e}, y_max={y_max:.10e}")
    print("# Requested / snapped Y ranges:")
    print(f"#  FUSE: [{y_f0:.10e}, {y_f1:.10e}] -> snapped [{y_f0s:.10e}, {y_f1s:.10e}]")
    print(f"#  WING: [{y_w0:.10e}, {y_w1:.10e}] -> snapped [{y_w0s:.10e}, {y_w1s:.10e}]")
    print("# Degrees:")
    print(f"#  FUSE: deg = ({degx_fuse}, {DEG_Y_FUSE}, {DEG_Z})")
    print(f"#  WING: deg = ({degx_wing}, {DEG_Y_WING}, {DEG_Z})")
    print("# FUSE padding:")
    print(f"#  PAD_LE={PAD_LE_FUSE}, PAD_TE={PAD_TE_FUSE}, PAD_Z={PAD_Z_FUSE}")
    print("# WING padding (LE/TE/Z root->tip):")
    print(f"#  PAD_LE: {PAD_LE_WING_ROOT} -> {PAD_LE_WING_TIP}")
    print(f"#  PAD_TE: {PAD_TE_WING_ROOT} -> {PAD_TE_WING_TIP}")
    print(f"#  PAD_Z : {PAD_Z_WING_ROOT} -> {PAD_Z_WING_TIP}")
    print("# --------------------------------------------\n")

    print(f"FFD_DEGREE = ({degx_fuse}, {DEG_Y_FUSE}, {DEG_Z})")
    print("FFD_DEFINITION = \\")
    print(
        f"  ({box_fuse}, "
        f"{fmt_pt(corners_fuse[0])}, {fmt_pt(corners_fuse[1])}, {fmt_pt(corners_fuse[2])}, {fmt_pt(corners_fuse[3])}, "
        f"{fmt_pt(corners_fuse[4])}, {fmt_pt(corners_fuse[5])}, {fmt_pt(corners_fuse[6])}, {fmt_pt(corners_fuse[7])})"
    )
    print("")

    print(f"FFD_DEGREE = ({degx_wing}, {DEG_Y_WING}, {DEG_Z})")
    print("FFD_DEFINITION = \\")
    print(
        f"  ({box_wing}, "
        f"{fmt_pt(corners_wing[0])}, {fmt_pt(corners_wing[1])}, {fmt_pt(corners_wing[2])}, {fmt_pt(corners_wing[3])}, "
        f"{fmt_pt(corners_wing[4])}, {fmt_pt(corners_wing[5])}, {fmt_pt(corners_wing[6])}, {fmt_pt(corners_wing[7])})"
    )
    print("")

    # -----------------------------
    # set_ffd_design_var.py calls
    # -----------------------------
    print("\n# --------------------------------------------")
    print("# set_ffd_design_var.py calls (ONE per box)")
    print("# --------------------------------------------\n")

    # NOTE: Here you are passing degrees (not deg+1). Keep this consistent with your
    # set_ffd_design_var.py expectation.
    run_set_ffd_for_box(box_fuse, degx_fuse, DEG_Y_FUSE, DEG_Z, run=RUN_SET_FFD)
    run_set_ffd_for_box(box_wing, degx_wing, DEG_Y_WING, DEG_Z, run=RUN_SET_FFD)

    # -----------------------------
    # DRAW
    # -----------------------------
    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=MESH_OPACITY, show_edges=SHOW_MESH_EDGES)

    all_ctrl_pts = []

    if DRAW_BOXES:
        pl.add_mesh(hexa_wireframe(corners_fuse), color="black", line_width=2)
        pl.add_mesh(hexa_wireframe(corners_wing), color="black", line_width=2)

        if DRAW_CTRL_POINTS:
            ctrl_f = trilinear_ctrl_points(corners_fuse, degx_fuse, DEG_Y_FUSE, DEG_Z)
            ctrl_w = trilinear_ctrl_points(corners_wing, degx_wing, DEG_Y_WING, DEG_Z)
            all_ctrl_pts.append(ctrl_f.reshape(-1, 3))
            all_ctrl_pts.append(ctrl_w.reshape(-1, 3))

    if DRAW_CTRL_POINTS and all_ctrl_pts:
        ctrl_pts = np.vstack(all_ctrl_pts)
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
