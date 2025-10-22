import numpy as np
import matplotlib.pyplot as plt
from defineMetrix import M_matrix

def M_cascade(tau, Z_list, omega):
    """Total transfer matrix at a given omega: Π_n M_n(omega, tau, Z_n)."""
    M = np.eye(2, dtype=complex)
    for Z in Z_list:
        M = M_matrix(Z, tau, omega) @ M
    return M

def vout_over_vin(omega_array, Z_list, tau, Vin=1.0, Iin=0.0):
    """
    For each omega, compute [Vout; Iout] = M_total(omega) * [Vin; Iin].
    Returns |Vout / Vin|.
    """
    Vin = complex(Vin); Iin = complex(Iin)
    mags = np.empty_like(omega_array, dtype=float)
    for i, w in enumerate(omega_array):
        Mtot = M_cascade(tau, Z_list, w)
        Vout = (Mtot @ np.array([Vin, Iin], dtype=complex))[0]
        mags[i] = np.abs(Vout / Vin)
    return mags

if __name__ == "__main__":
    # --- Parameters ---
    Z_list = [50.0, 93.0, 50.0, 93.0, 93.0, 50.0]
    d = 0.5                     # section length (m)
    c = 3e8                     # wave speed (m/s)
    tau = d / c                 # time per section

    # Frequency sweep (0 -> 300 MHz)
    freq = np.linspace(0, 300e6, 20000)   # Hz
    omega = 2 * np.pi * freq

    # Compute magnitude
    mag = vout_over_vin(omega, Z_list, tau, Vin=1.0, Iin=0.0)

    # --- Plot ---
    plt.figure(figsize=(9, 5))
    plt.plot(freq / 1e6, mag, linewidth=1.1)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|V_{\mathrm{out}}/V_{\mathrm{in}}|$")
    plt.title(r"Fixed $|V_{\mathrm{out}}/V_{\mathrm{in}}|$ vs Frequency up to 300 MHz")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
