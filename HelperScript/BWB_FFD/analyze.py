#!/usr/bin/env python3
import numpy as np
import pyvista as pv

# -----------------------------
# Inputs
# -----------------------------
VTU_FILE = "surface_deformed.vtu"

# Requested global degrees across the wing
DEG_X, DEG_Y, DEG_Z = 8, 15, 2
N_BOXES = DEG_Y
NY_STATIONS = N_BOXES + 1  # 16 stations => 15 spanwise boxes

# Padding relative to local chord and local z-thickness
PAD_LE = 0.03   # fraction of local chord ahead of LE
PAD_TE = 0.05   # fraction of local chord behind TE
PAD_Z  = 0.15   # fraction of local z-thickness above/below

# Spanwise neighborhood for estimating LE/TE and z extents
WINDOW_FRAC = 1.5

# Robust percentiles (avoid stray outliers)
LE_PCTL   = 1.0
TE_PCTL   = 99.0
ZLO_PCTL  = 1.0
ZHI_PCTL  = 99.0

BOX_PREFIX = "WING"

# Visualization toggles
SHOW_MESH_EDGES = False
MESH_OPACITY = 0.60
DRAW_BOXES = True
DRAW_BOX_POINTS = False


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
    """
    Build a wireframe (polyline) representation of a hexahedron from 8 corners.
    Corners are assumed to be [p1..p8] in the order above.
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = corners
    edges = [
        (p1, p2), (p2, p3), (p3, p4), (p4, p1),  # bottom
        (p5, p6), (p6, p7), (p7, p8), (p8, p5),  # top
        (p1, p5), (p2, p6), (p3, p7), (p4, p8),  # verticals
    ]

    # Create one PolyData with multiple line segments
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
    """
    Generate control points inside an SU2 hexahedral FFD box
    using trilinear interpolation of the 8 corners.

    corners: list of 8 points [p1..p8] in SU2 order
    returns array of shape (nx, ny, nz, 3)
    """
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
    zth   = np.maximum(zhi - zlo, 1e-12)

    # Tapered bounds with padding (station-wise)
    xmin = xLE - PAD_LE * chord
    xmax = xLE + (1.0 + PAD_TE) * chord
    zmin = zlo - PAD_Z * zth
    zmax = zhi + PAD_Z * zth

    # -----------------------------
    # PRINT SU2 SNIPPET (stdout)
    # -----------------------------
    print("\n# --------------------------------------------")
    print("# SU2 FFD: tapered spanwise boxes from VTU")
    print("# Axes: chord=X, span=Y, dihedral/thickness=Z")
    print("# Requested global degrees: (8, 15, 2)")
    print("# Implemented as 15 boxes with per-box degree: (8, 1, 2)")
    print("# --------------------------------------------\n")

    print(f"FFD_DEGREE = ({DEG_X}, 1, {DEG_Z})\n")

    defs = []
    all_boxes_corners = []

    for j in range(N_BOXES):
        y0, y1 = y_st[j], y_st[j + 1]

        corners = su2_hexa_corners(
            xmin[j], xmax[j], y0, zmin[j], zmax[j],
            xmin[j+1], xmax[j+1], y1, zmin[j+1], zmax[j+1]
        )
        all_boxes_corners.append(corners)

        name = f"{BOX_PREFIX}_{j:03d}"
        line = (
            f"({name}, "
            f"{fmt_pt(corners[0])}, {fmt_pt(corners[1])}, {fmt_pt(corners[2])}, {fmt_pt(corners[3])}, "
            f"{fmt_pt(corners[4])}, {fmt_pt(corners[5])}, {fmt_pt(corners[6])}, {fmt_pt(corners[7])})"
        )
        defs.append(line)

    print("FFD_DEFINITION = \\")
    print("  " + ";\n  ".join(defs))
    print("\n# Diagnostics (root/mid/tip):")
    for idx in [0, NY_STATIONS // 2, NY_STATIONS - 1]:
        print(f"#  y={y_st[idx]: .6e}, xLE~{xLE[idx]: .6e}, chord~{chord[idx]: .6e}, "
              f"zmin~{zmin[idx]: .6e}, zmax~{zmax[idx]: .6e}")

    # -----------------------------
    # DRAW (PyVista)
    # -----------------------------
    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=MESH_OPACITY, show_edges=SHOW_MESH_EDGES)

    all_ctrl_pts = []

    if DRAW_BOXES:
        for corners in all_boxes_corners:
            # Draw FFD box
            wf = hexa_wireframe(corners)
            pl.add_mesh(wf, color="black", line_width=2)

            # Generate & collect control points
            ctrl = trilinear_ctrl_points(corners, DEG_X, 1, DEG_Z)
            all_ctrl_pts.append(ctrl.reshape(-1, 3))

    # Plot control points
    if len(all_ctrl_pts) > 0:
        ctrl_pts = np.vstack(all_ctrl_pts)
        pl.add_points(
            ctrl_pts,
            render_points_as_spheres=True,
            point_size=10,
            color="red",
        )

    pl.add_axes()
    pl.show_grid()
    pl.show()



if __name__ == "__main__":
    main()
