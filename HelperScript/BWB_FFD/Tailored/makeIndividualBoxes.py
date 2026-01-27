#!/usr/bin/env python3
import numpy as np
import pyvista as pv
import subprocess

# -----------------------------
# Inputs
# -----------------------------
VTU_FILE = "surface_flow.vtu"

# Spanwise segmentation (15 boxes => 16 stations)
DEG_Y = 15
N_BOXES = DEG_Y
NY_STATIONS = N_BOXES + 1

# Vertical degree (SU2 convention: nk = DEG_Z + 1 control points)
DEG_Z = 1

# Option 2: variable deg_x per spanwise box
MIN_DEG_X = 3
MAX_DEG_X = 8

# Choose how deg_x is selected:
#   - "max_chord": target_dx = max_chord / (MAX_DEG_X+1)
#   - "absolute":  target_dx = TARGET_DX_ABS (in your mesh units)
TARGET_DX_MODE = "max_chord"
TARGET_DX_ABS = None  # e.g. 0.05 if TARGET_DX_MODE == "absolute"

# Padding relative to local chord and local z-thickness
PAD_LE = 0.03   # fraction of local chord ahead of LE
PAD_TE = 0.05   # fraction of local chord behind TE
PAD_Z  = 0.15   # fraction of local z-thickness above/below

# Spanwise neighborhood for estimating LE/TE and z extents
WINDOW_FRAC = 1.5

# Robust percentiles (avoid stray outliers)
LE_PCTL  = 1.0
TE_PCTL  = 99.0
ZLO_PCTL = 1.0
ZHI_PCTL = 99.0

BOX_PREFIX = "WING"

# -----------------------------
# set_ffd_design_var.py (ONE call per box)
# -----------------------------
RUN_SET_FFD = False  # True => actually run; False => dry-run print
SET_FFD_SCRIPT = "set_ffd_design_var.py"
MARKERS = "'wing_top, wing_bottom, tip_top, tip_bottom, TE, tip_TE'"

# Visualization toggles
SHOW_MESH_EDGES = False
MESH_OPACITY = 0.60
DRAW_BOXES = True
DRAW_CTRL_POINTS = True
CTRL_POINT_SIZE = 10

# -----------------------------
# Tecplot outputs
# -----------------------------
WRITE_TEC_CTRL_DAT = True
TEC_CTRL_DAT_FILE = f"{BOX_PREFIX}_ffd_ctrl_points.dat"

WRITE_TEC_BOX_WIREFRAME_DAT = True
TEC_BOX_DAT_FILE = f"{BOX_PREFIX}_ffd_boxes_wireframe.dat"


def spanwise_profiles(points, y_stations, half_window):
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
            mask = np.ones_like(y, dtype=bool)

        xs = x[mask]
        zs = z[mask]

        xLE[j] = np.percentile(xs, LE_PCTL)
        xTE[j] = np.percentile(xs, TE_PCTL)
        zlo[j] = np.percentile(zs, ZLO_PCTL)
        zhi[j] = np.percentile(zs, ZHI_PCTL)

    xTE = np.maximum(xTE, xLE + 1e-12)
    return xLE, xTE, zlo, zhi


def choose_deg_x_per_box(chord_box, min_deg=3, max_deg=8, mode="max_chord", target_dx_abs=None):
    if mode == "absolute":
        if target_dx_abs is None or target_dx_abs <= 0:
            raise ValueError("TARGET_DX_ABS must be set (>0) when TARGET_DX_MODE='absolute'")
        target_dx = float(target_dx_abs)
    elif mode == "max_chord":
        target_dx = chord_box.max() / (max_deg + 1)
    else:
        raise ValueError("TARGET_DX_MODE must be 'max_chord' or 'absolute'")

    nctrl = np.clip(np.round(chord_box / target_dx).astype(int), min_deg + 1, max_deg + 1)
    deg_x = (nctrl - 1).astype(int)
    return deg_x


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


def hexa_wireframe_edges(corners):
    p1, p2, p3, p4, p5, p6, p7, p8 = corners
    return [
        (p1, p2), (p2, p3), (p3, p4), (p4, p1),
        (p5, p6), (p6, p7), (p7, p8), (p8, p5),
        (p1, p5), (p2, p6), (p3, p7), (p4, p8),
    ]


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
                    (1 - x) * (1 - y) * (1 - z) * p1 +
                    x * (1 - y) * (1 - z) * p2 +
                    x * y * (1 - z) * p3 +
                    (1 - x) * y * (1 - z) * p4 +
                    (1 - x) * (1 - y) * z * p5 +
                    x * (1 - y) * z * p6 +
                    x * y * z * p7 +
                    (1 - x) * y * z * p8
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


def write_tecplot_ctrl_points_xyz(filename, title, xyz):
    """
    Tecplot ASCII point cloud using FELINESEG with degenerate segments.
    Variables: X Y Z only.
    """
    xyz = np.asarray(xyz, dtype=float)
    n = xyz.shape[0]

    with open(filename, "w") as f:
        f.write(f'TITLE = "{title}"\n')
        f.write('VARIABLES = "X", "Y", "Z"\n')
        f.write(f'ZONE T="{title}", N={n}, E={n}, ZONETYPE=FELINESEG, DATAPACKING=POINT\n')

        # coordinates
        for i in range(n):
            x, y, z = xyz[i]
            f.write(f"{x:.15e} {y:.15e} {z:.15e}\n")

        # degenerate line segments (i -> i)
        for i in range(1, n + 1):
            f.write(f"{i} {i}\n")

    print(f"# Wrote Tecplot XYZ control points: {filename} (N={n})")





def write_tecplot_line_zone(filename, title, segments):
    """
    Write Tecplot ASCII .dat with a single line zone using FE LINES.
    segments: list of (pA, pB) with pA/pB shape (3,)
    """
    # Build point list + connectivity
    pts = []
    conn = []
    idx = 1  # Tecplot connectivity is 1-based
    for a, b in segments:
        pts.append(a); pts.append(b)
        conn.append((idx, idx + 1))
        idx += 2

    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    e = len(conn)

    with open(filename, "w") as f:
        f.write(f'TITLE = "{title}"\n')
        f.write('VARIABLES = "X", "Y", "Z"\n')
        f.write(f'ZONE T="{title}", N={n}, E={e}, ZONETYPE=FELINESEG, DATAPACKING=POINT\n')
        for p in pts:
            f.write(f"{p[0]:.15e} {p[1]:.15e} {p[2]:.15e}\n")
        for (i1, i2) in conn:
            f.write(f"{i1} {i2}\n")

    print(f"# Wrote Tecplot line zone: {filename}  (segments={e})")


def main():
    mesh = pv.read(VTU_FILE)
    pts = mesh.points

    # Span stations
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    y_st = np.linspace(y_min, y_max, NY_STATIONS)

    span = y_max - y_min
    dy = span / max(NY_STATIONS - 1, 1)
    half_window = 0.5 * WINDOW_FRAC * dy

    # Profiles
    xLE, xTE, zlo, zhi = spanwise_profiles(pts, y_st, half_window)
    chord = np.maximum(xTE - xLE, 1e-12)
    zth = np.maximum(zhi - zlo, 1e-12)

    # Tapered bounds with padding (station-wise)
    xmin = xLE - PAD_LE * chord
    xmax = xLE + (1.0 + PAD_TE) * chord
    zmin = zlo - PAD_Z * zth
    zmax = zhi + PAD_Z * zth

    # Representative chord per BOX (avg of end stations)
    chord_box = 0.5 * (chord[:-1] + chord[1:])  # length N_BOXES

    # Variable deg_x per box
    deg_x_box = choose_deg_x_per_box(
        chord_box,
        min_deg=MIN_DEG_X,
        max_deg=MAX_DEG_X,
        mode=TARGET_DX_MODE,
        target_dx_abs=TARGET_DX_ABS
    )

    # Build corners for each box
    all_boxes_corners = []
    for j in range(N_BOXES):
        y0, y1 = y_st[j], y_st[j + 1]
        corners = su2_hexa_corners(
            xmin[j], xmax[j], y0, zmin[j], zmax[j],
            xmin[j + 1], xmax[j + 1], y1, zmin[j + 1], zmax[j + 1]
        )
        all_boxes_corners.append(corners)

    # -----------------------------
    # PRINT SU2 blocks
    # -----------------------------
    print("\n# --------------------------------------------")
    print("# SU2 FFD: tapered spanwise boxes from VTU")
    print("# Axes: chord=X, span=Y, dihedral/thickness=Z")
    print(f"# Spanwise boxes: {N_BOXES} (stations={NY_STATIONS})")
    print(f"# Per-box degrees: (deg_x[j], 1, {DEG_Z})  with deg_x varying by chord")
    print("# --------------------------------------------\n")

    print("# deg_x per box (j : deg_x, chord_box):")
    for j in range(N_BOXES):
        print(f"#  {j:02d} : {deg_x_box[j]}   (c~{chord_box[j]:.6e})")

    print("\n# --------------------------------------------")
    print("# SU2 FFD boxes (sequential, per-box degree)")
    print("# --------------------------------------------\n")

    for j in range(N_BOXES):
        degx = int(deg_x_box[j])
        degy = 1
        degz = int(DEG_Z)

        box_name = f"{BOX_PREFIX}_{j:03d}"
        corners = all_boxes_corners[j]

        print(f"# Box {j:02d}: {box_name}")
        print(f"FFD_DEGREE = ({degx}, {degy}, {degz})")
        print("FFD_DEFINITION = \\")
        print(
            "  "
            f"({box_name}, "
            f"{fmt_pt(corners[0])}, {fmt_pt(corners[1])}, "
            f"{fmt_pt(corners[2])}, {fmt_pt(corners[3])}, "
            f"{fmt_pt(corners[4])}, {fmt_pt(corners[5])}, "
            f"{fmt_pt(corners[6])}, {fmt_pt(corners[7])})\n"
        )

    print("\n# --------------------------------------------")
    print("# set_ffd_design_var.py calls (ONE per box)")
    print("# --------------------------------------------\n")

    for box_j in range(N_BOXES):
        degx = int(deg_x_box[box_j])
        degy = 1
        degz = int(DEG_Z)

        ni = degx
        nj = degy
        nk = degz

        box_name = f"{BOX_PREFIX}_{box_j:03d}"
        run_set_ffd_for_box(box_name, ni, nj, nk, run=RUN_SET_FFD)

    # -----------------------------
    # Collect control points + wireframe segments for Tecplot outputs
    # -----------------------------
    all_ctrl_pts = []
    all_ctrl_box = []
    all_ctrl_degx = []

    all_segments = []  # box wireframe segments

    for j, corners in enumerate(all_boxes_corners):
        if DRAW_CTRL_POINTS:
            ctrl = trilinear_ctrl_points(corners, int(deg_x_box[j]), 1, DEG_Z)
            pts_j = ctrl.reshape(-1, 3)
            all_ctrl_pts.append(pts_j)
            all_ctrl_box.append(np.full(pts_j.shape[0], j, dtype=int))
            all_ctrl_degx.append(np.full(pts_j.shape[0], int(deg_x_box[j]), dtype=int))

        if DRAW_BOXES:
            all_segments.extend(hexa_wireframe_edges(corners))

    if len(all_ctrl_pts) > 0:
        ctrl_pts = np.vstack(all_ctrl_pts)
        ctrl_box = np.concatenate(all_ctrl_box)
        ctrl_degx = np.concatenate(all_ctrl_degx)

        if WRITE_TEC_CTRL_DAT:
            write_tecplot_ctrl_points_xyz(
                "ffd_ctrl_points.dat",
                "FFD Control Points",
                ctrl_pts
            )

    if WRITE_TEC_BOX_WIREFRAME_DAT and len(all_segments) > 0:
        write_tecplot_line_zone(
            TEC_BOX_DAT_FILE,
            title=f"{BOX_PREFIX} FFD Boxes Wireframe",
            segments=all_segments,
        )

    # -----------------------------
    # DRAW (PyVista)
    # -----------------------------
    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=MESH_OPACITY, show_edges=SHOW_MESH_EDGES)

    if DRAW_BOXES:
        for corners in all_boxes_corners:
            # show as black wireframe using line segments
            for a, b in hexa_wireframe_edges(corners):
                pl.add_lines(np.vstack([a, b]), color="black", width=2)

    if DRAW_CTRL_POINTS and len(all_ctrl_pts) > 0:
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
