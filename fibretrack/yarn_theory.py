"""Classical twisted-yarn models as testable functions.

All lengths in um, twist T in turns per metre, angles in degrees unless noted.
Polar angle phi is counter-clockwise as displayed (u = x - xc, v = -(y - yc)).

References
  Hearle, Grosberg & Backer (1969) Structural Mechanics of Fibers, Yarns and Fabrics - ideal helical yarn
  Treloar (1956) - twist contraction
  Morton (1956) Text. Res. J. 26:325 - fibre migration
  Hearle, Gupta & Merchant (1965) Text. Res. J. 35:329 - migration parameters Ybar, D, I
  Hamilton (1958) J. Text. Inst. 49:T411 - radial distribution (migration) index of blends
"""
import numpy as np


# ------------------------------------------------------------ ideal helix
def helix_angle(r_um, T_tpm):
    """ideal helix angle at radius r for twist T: tan(theta) = 2 pi r T"""
    return np.degrees(np.arctan(2 * np.pi * np.asarray(r_um) * 1e-6 * T_tpm))


def fit_helix_twist(r_um, tan_tangential, rmin=30.0):
    """least-squares T (through origin) from tangential slopes; returns T, residual SD (deg), corr"""
    r = np.asarray(r_um, float); t = np.asarray(tan_tangential, float)
    m = np.isfinite(r) & np.isfinite(t) & (r > rmin)
    b = np.sum(t[m] * r[m]) / np.sum(r[m] ** 2)
    resid = np.degrees(np.arctan(t[m]) - np.arctan(b * r[m]))
    return b / (2 * np.pi) * 1e6, float(np.std(resid)), float(np.corrcoef(r[m], t[m])[0, 1])


def binned_median(x, y, bins):
    k = np.digitize(x, bins)
    return np.array([[0.5 * (bins[j - 1] + bins[j]), np.median(y[k == j])] for j in range(1, len(bins)) if (k == j).sum() > 20])


# ------------------------------------------------------------ twist from positions
def rigid_rotation_deg(P0, P1):
    """least-squares rotation (deg, CCW on screen) taking point set P0 to P1; rows (y, x)"""
    a0 = P0 - P0.mean(0); a1 = P1 - P1.mean(0)
    u0, v0, u1, v1 = a0[:, 1], -a0[:, 0], a1[:, 1], -a1[:, 0]
    return float(np.degrees(np.arctan2(np.sum(u0 * v1 - v0 * u1), np.sum(u0 * u1 + v0 * v1))))


def twist_from_rotation(angle_deg, dz_um):
    return angle_deg / 360.0 / (dz_um * 1e-6)


def twist_contraction(theta_surface_deg):
    """Treloar-type yarn contraction for an ideal helix (mean cos over the cross-section, uniform packing)"""
    t = np.radians(theta_surface_deg)
    return 1 - 2 * (1 - np.cos(t)) / (np.tan(t) ** 2) if t > 0 else 0.0


# ------------------------------------------------------------ migration
def migration_parameters(r_um, z_um, R_um):
    """Hearle-Gupta-Merchant: mean fibre position Ybar, RMS deviation D, migration intensity I (per mm)
    for one fibre; Y = (r/R)^2. Ideal complete migration: Ybar 0.5, D = 1/sqrt(12) = 0.289."""
    Y = (np.asarray(r_um) / np.asarray(R_um)) ** 2
    dY = np.gradient(Y, np.asarray(z_um)) * 1000.0
    return float(Y.mean()), float(Y.std()), float(np.sqrt(np.mean(dY ** 2)))


def hamilton_index(r, is_A, R):
    """Radial distribution index of component A in a two-component assembly (Hamilton 1958 idea):
    +100 = A wholly in the outer annulus it could fill, 0 = uniformly blended, -100 = wholly central.
    r: radii of all fibre-phase samples (pixels or fibre centres), is_A: bool, R: yarn radius."""
    Y = (np.asarray(r) / R) ** 2
    yA, yall, fA = Y[is_A].mean(), Y.mean(), np.mean(is_A)
    ymax, ymin = 1 - fA / 2.0, fA / 2.0
    return float(100 * (yA - yall) / ((ymax - yall) if yA >= yall else (yall - ymin)))


def bending_rigidity_ratio(E1_GPa, d1_um, E2_GPa, d2_um):
    """single-fibre bending rigidity ratio EI ~ E d^4 (component 1 / component 2)"""
    return (E1_GPa * d1_um ** 4) / (E2_GPa * d2_um ** 4)
