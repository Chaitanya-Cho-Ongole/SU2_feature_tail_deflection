#!/usr/bin/env python3
"""
Speed of sound (ISA up to 86 km), velocity from Mach, viscosity, pressure, density,
and optional Reynolds number.

Usage:
    python speed_of_sound.py --altitude 10000 --mach 0.78 --length 1.5
"""

import argparse
import math

# --- Constants ---
G0 = 9.80665       # gravity [m/s^2]
R  = 287.05        # specific gas constant for dry air [J/(kg·K)]
GAMMA = 1.4        # cp/cv for dry air
P0 = 101325.0      # sea-level standard pressure [Pa]
T0 = 288.15        # sea-level standard temperature [K]

# ISA layers: (base_height [m], lapse_rate L [K/m])
LAYERS = [
    (0.0,     -0.0065),  # 0–11 km (troposphere)
    (11000.0,  0.0),     # 11–20 km (isothermal)
    (20000.0,  0.001),   # 20–32 km
    (32000.0,  0.0028),  # 32–47 km
    (47000.0,  0.0),     # 47–51 km (isothermal)
    (51000.0, -0.0028),  # 51–71 km
    (71000.0, -0.002),   # 71–86 km
]

def temperature_at_altitude(h: float) -> float:
    """ISA temperature T[h] [K] up to ~86 km."""
    T = T0
    h_base = 0.0
    for h_layer, L in LAYERS:
        if h <= h_layer:
            return T + (h - h_base) * L
        T += (h_layer - h_base) * L
        h_base = h_layer
    # simple linear extrapolation above the last layer
    return T + (h - h_base) * LAYERS[-1][1]

def pressure_density_at_altitude(h: float):
    """
    ISA pressure p [Pa] and density rho [kg/m^3] up to ~86 km.
    Integrates hydrostatic equation across layers.
    """
    T = T0
    p = P0
    h_base = 0.0

    for h_layer, L in LAYERS:
        if h <= h_layer:
            # integrate only to the requested altitude
            dh = h - h_base
            if abs(L) < 1e-12:  # isothermal segment
                p *= math.exp(-G0 * dh / (R * T))
                T_end = T  # no change in T
            else:            # gradient segment
                T_end = T + L * dh
                p *= (T_end / T) ** (-G0 / (L * R))
            T = T_end
            rho = p / (R * T)
            return p, rho, T

        # integrate full segment to the layer top
        dh = h_layer - h_base
        if abs(L) < 1e-12:
            p *= math.exp(-G0 * dh / (R * T))
            # T unchanged
        else:
            T_top = T + L * dh
            p *= (T_top / T) ** (-G0 / (L * R))
            T = T_top
        h_base = h_layer

    # above last layer: continue with last lapse rate
    L_last = LAYERS[-1][1]
    dh = h - h_base
    if abs(L_last) < 1e-12:
        p *= math.exp(-G0 * dh / (R * T))
        # T unchanged
    else:
        T_top = T + L_last * dh
        p *= (T_top / T) ** (-G0 / (L_last * R))
        T = T_top
    rho = p / (R * T)
    return p, rho, T

def speed_of_sound(h: float) -> float:
    """Speed of sound a [m/s] at altitude h [m]."""
    # Use T from ISA layers to keep exactly consistent with p, rho if needed
    T = temperature_at_altitude(h)
    return math.sqrt(GAMMA * R * T)

def dynamic_viscosity(T: float) -> float:
    """Dynamic viscosity μ [Pa·s] via Sutherland's law."""
    mu0 = 1.716e-5  # Pa·s at T0suth
    T0s = 273.15    # K
    C = 110.4       # K
    return mu0 * (T / T0s) ** 1.5 * (T0s + C) / (T + C)

def velocity_from_mach(M: float, h: float):
    """True airspeed from Mach at altitude. Returns (m/s, km/h, kt)."""
    if M < 0:
        raise ValueError("Mach number must be non-negative.")
    a = speed_of_sound(h)
    V_ms = M * a
    return V_ms, V_ms * 3.6, V_ms * 1.94384449244

def main():
    p = argparse.ArgumentParser(description="ISA a, V(Mach), μ, ν, p, ρ, and optional Reynolds number.")
    p.add_argument("--altitude", type=float, required=True, help="Altitude in meters")
    p.add_argument("--mach", type=float, required=True, help="Mach number (e.g., 0.78)")
    p.add_argument("--length", type=float, default=None,
                   help="Characteristic length in meters (for Reynolds number)")
    args = p.parse_args()

    # Thermo @ altitude
    p_Pa, rho, T = pressure_density_at_altitude(args.altitude)
    a = math.sqrt(GAMMA * R * T)  # consistent with the same T used in p, rho
    mu = dynamic_viscosity(T)
    nu = mu / rho  # kinematic viscosity [m^2/s]

    # Velocity from Mach
    V_ms, V_kmh, V_kts = velocity_from_mach(args.mach, args.altitude)

    print(f"Altitude:            {args.altitude:.1f} m")
    print(f"Temperature:         {T:.2f} K")
    print(f"Pressure:            {p_Pa:.2f} Pa")
    print(f"Density:             {rho:.6f} kg/m^3")
    print(f"Speed of sound:      {a:.2f} m/s")
    print(f"Mach:                {args.mach:.3f}")
    print(f"True airspeed:       {V_ms:.2f} m/s  |  {V_kmh:.2f} km/h  |  {V_kts:.2f} kt")
    print(f"Dynamic viscosity μ: {mu:.6e} Pa·s")
    print(f"Kinematic visc.  ν:  {nu:.6e} m^2/s")

    if args.length is not None:
        Re = rho * V_ms * args.length / mu
        print(f"Reynolds number Re (L={args.length:g} m): {Re:.3e}")

if __name__ == "__main__":
    main()
