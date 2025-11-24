# -*- coding: utf-8 -*-
"""
Transmission-line network → Tight-binding form with ports.
Implements Eq. (24): (H + iΓ)v = ε v + iV  ⇒  v = i (H + iΓ - ε I)^(-1) V
and Eq. (31): v_out = i G_{β α} √(1-ε²) v_in  →  T = v_out / v_in

Now:
  ε = cos(ωτ),  ω = 2πf,  τ = π / (2ω0),  f0 = 114 MHz
  Sweep f ∈ [0, 600 MHz] and plot |T_phys| vs f.
"""

import numpy as np
import random
import matplotlib.pyplot as plt


# ---------- Core Builders ----------
def build_H_and_Gamma(n, connections, Z_links, ports_in=None, ports_out=None):
    ports_in = ports_in or {}
    ports_out = ports_out or {}

    # If Z_links is a list, map each connection to its impedance
    if isinstance(Z_links, (list, tuple, np.ndarray)):
        Z_links_dict = {connections[k]: float(Z_links[k]) for k in range(len(connections))}
        Z_links = Z_links_dict

    # Z^-1 adjacency matrix
    Zinv = np.zeros((n, n), dtype=float)
    def get_Z(i, j):
        if isinstance(Z_links, (int, float)):
            return float(Z_links)
        return Z_links.get((i, j), Z_links.get((j, i), None))

    for (i, j) in connections:
        Zij = get_Z(i, j)
        if Zij is None or Zij == 0:
            raise ValueError(f"No impedance for link {(i, j)}")
        val = 1.0 / Zij
        Zinv[i, j] = Zinv[j, i] = val

    # σ_n = 1 / sqrt(sum_j Z^-1_ij)
    sigma = np.zeros(n, dtype=float)
    for i in range(n):
        ssum = np.sum(Zinv[i])
        sigma[i] = 1.0 / np.sqrt(ssum) if ssum > 0 else 0.0

    # H_ij = σ_i σ_j Z^-1_ij (no onsite terms)
    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if i != j and Zinv[i, j] != 0.0:
                H[i, j] = sigma[i] * sigma[j] * Zinv[i, j]

    # Γ_diag accumulates input/output ports: Γ_n = σ_n^2 / Z_port
    Gamma_diag = np.zeros(n, dtype=float)
    for i, Zin in ports_in.items():
        Gamma_diag[i] += (sigma[i] ** 2) / Zin
    for j, Zout in ports_out.items():
        Gamma_diag[j] += (sigma[j] ** 2) / Zout
    Gamma = np.diag(Gamma_diag)

    return H, Gamma, sigma


def make_A(H, Gamma, eps):
    """A = H + i Γ_s - ε I, where Γ_s = Γ * sqrt(1 - ε^2)."""
    s = np.sqrt(max(0.0, 1.0 - eps**2))
    Gamma_s = Gamma * s
    I = np.eye(H.shape[0], dtype=complex)
    A = H + 1j * Gamma_s - eps * I
    return A, Gamma_s


def build_driving_vector(n, sigma, ports_in, eps, Vin=1.0):
    """V_α = √(1 - ε²) * σ_α / Z_in * Vin."""
    s = np.sqrt(max(0.0, 1.0 - eps**2))
    V = np.zeros(n, dtype=complex)
    for α, Zin in (ports_in or {}).items():
        vin = sigma[α] / Zin * Vin
        V[α] += s * vin
    return V


def compute_T_phys(H, Gamma, sigma, ports_in, ports_out, eps, Vin=1.0):
    """Compute physical transmission T_phys for a given ε."""
    A, _ = make_A(H, Gamma, eps)
    G = np.linalg.inv(A)
    V = build_driving_vector(len(sigma), sigma, ports_in, eps, Vin=Vin)
    v = 1j * (G @ V)
    α = list(ports_in.keys())[0]
    β = list(ports_out.keys())[0]
    T_phys = (sigma[β] * v[β]) / (sigma[α] * v[α])
    return T_phys


# ---------- Main Frequency Sweep ----------
if __name__ == "__main__":
    n = 5
    connections = [(0,1), (1,2), (2,3), (3,4)]

    # choose impedance pattern
    mode = "periodic"   # "periodic" or "random"
    if mode == "periodic":
        Z_list = [50 if i % 2 == 0 else 93 for i in range(len(connections))]
    elif mode == "random":
        Z_list = [random.choice([50, 93]) for _ in range(len(connections))]
    else:
        Z_list = [50.0 for _ in range(len(connections))]

    ports_in  = {0: 50.0}
    ports_out = {4: 50.0}
    Vin = 1.0

    # Build constant parts
    H, Gamma, sigma = build_H_and_Gamma(n, connections, Z_list, ports_in, ports_out)

    # Frequency sweep
    f0 = 114e6
    omega0 = 2 * np.pi * f0
    tau = np.pi / (2.0 * omega0)

    f_range = np.linspace(0, 600e6, 601)
    T_vals = []

    for f in f_range:
        omega = 2 * np.pi * f
        eps = np.cos(omega * tau)
        T_phys = compute_T_phys(H, Gamma, sigma, ports_in, ports_out, eps)
        T_vals.append(abs(T_phys))

    # Plot |T_phys| vs frequency
    plt.figure(figsize=(8, 4))
    plt.plot(f_range / 1e6, T_vals, lw=1.8)
    plt.title(f"|T_phys| vs Frequency (Z_list={Z_list})")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("|T_phys|")
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
