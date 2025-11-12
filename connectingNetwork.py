# -*- coding: utf-8 -*-
"""
Transmission-line network → Tight-binding form with ports.
Implements Eq. (24): (H + iΓ)v = ε v + iV  ⇒  v = i (H + iΓ - ε I)^(-1) V
and Eq. (31): v_out = i G_{β α} √(1-ε²) v_in  →  T = v_out / v_in
"""

import numpy as np
import random


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

    return H, Gamma, sigma, Zinv


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


# ---------- Example Network ----------
if __name__ == "__main__":
    n = 5
    connections = [(0,1), (1,2), (2,3), (3,4)]

    # --- choose impedance pattern ---
    mode = "periodic"   # "periodic" or "random"

    if mode == "periodic":
        Z_list = [50 if i % 2 == 0 else 93 for i in range(len(connections))]
    elif mode == "random":
        Z_list = [random.choice([50, 93]) for _ in range(len(connections))]
    else:
        Z_list = [50.0 for _ in range(len(connections))]

    # Ports and parameters
    ports_in  = {0: 50.0}
    ports_out = {4: 50.0}
    eps = 0.3
    Vin = 1.0

    # --- build system ---
    H, Gamma, sigma, Zinv = build_H_and_Gamma(
        n, connections, Z_list, ports_in=ports_in, ports_out=ports_out
    )
    A, Gamma_s = make_A(H, Gamma, eps)
    G = np.linalg.inv(A)
    V = build_driving_vector(n, sigma, ports_in, eps, Vin=Vin)
    v = 1j * (G @ V)
    V_phys = sigma * v

    # --- input/output voltages ---
    α = list(ports_in.keys())[0]
    β = list(ports_out.keys())[0]
    v_in = v[α]
    v_out = v[β]
    T_phys = (sigma[β] * v[β]) / (sigma[α] * v[α])


    # --- print results ---
    np.set_printoptions(precision=6, suppress=True)
    print("Z_list (Ω) =", Z_list)
    print(f"\nInput site α={α}, Output site β={β}")
    print(f"v_in  = {v_in}")
    print(f"v_out = {v_out}")
    print(f"T = v_out / v_in = {T_phys}  (|T| = {abs(T_phys):.6f}, |T|² = {abs(T_phys)**2:.6f})")

    print("\n--- Matrices summary ---")
    print("H:\n", np.round(H, 6))
    print("\nΓ_s:\n", np.round(Gamma_s, 6))
    print("\nA = H + iΓ_s - εI:\n", np.round(A, 6))
    print("\nG = A⁻¹:\n", np.round(G, 6))
