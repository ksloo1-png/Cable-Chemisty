# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt

def build_H_and_Gamma(n, connections, Z_links, ports_in=None, ports_out=None):
    ports_in = ports_in or {}
    ports_out = ports_out or {}
    if isinstance(Z_links, (list, tuple, np.ndarray)):
        Z_links = {connections[k]: float(Z_links[k]) for k in range(len(connections))}
    Zinv = np.zeros((n, n), dtype=float)
    def get_Z(i, j):
        if isinstance(Z_links, (int, float)): return float(Z_links)
        return Z_links.get((i, j), Z_links.get((j, i), None))
    for (i, j) in connections:
        Zij = get_Z(i, j);  val = 1.0 / float(Zij)
        Zinv[i, j] = Zinv[j, i] = val
    sigma = np.zeros(n, dtype=float)
    for i in range(n):
        ssum = np.sum(Zinv[i])
        sigma[i] = 1.0 / np.sqrt(ssum) if ssum > 0 else 0.0
    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if i != j and Zinv[i, j] != 0.0:
                H[i, j] = sigma[i] * sigma[j] * Zinv[i, j]
    Gamma_diag = np.zeros(n, dtype=float)
    for i, Zin in (ports_in or {}).items():
        Gamma_diag[i] += (sigma[i] ** 2) / float(Zin)
    for j, Zout in (ports_out or {}).items():
        Gamma_diag[j] += (sigma[j] ** 2) / float(Zout)
    Gamma = np.diag(Gamma_diag)
    return H, Gamma, sigma

def make_A(H, Gamma, eps):
    s = np.sqrt(max(0.0, 1.0 - eps**2))
    return H + 1j * (Gamma * s) - eps * np.eye(H.shape[0], dtype=complex)

def build_driving_vector(n, sigma, ports_in, eps, Vin=1.0):
    s = np.sqrt(max(0.0, 1.0 - eps**2))
    V = np.zeros(n, dtype=complex)
    for a, Zin in (ports_in or {}).items():
        v_in_scaled = sigma[a] * (1.0 / float(Zin)) * float(Vin)
        V[a] += s * v_in_scaled
    return V

def T_voltage_ratio_vs_freq(f_range, connections, Z_list, ports_in, ports_out, n, Vin=1.0):
    H, Gamma, sigma = build_H_and_Gamma(n, connections, Z_list, ports_in, ports_out)
    a = list(ports_in.keys())[0];  b = list(ports_out.keys())[0]
    f0 = 114e6;  omega0 = 2*np.pi*f0;  tau = np.pi/(2.0*omega0)
    Tabs = []
    for f in f_range:
        eps = np.cos(2*np.pi*f * tau)
        A = make_A(H, Gamma, eps)
        G = np.linalg.inv(A)
        Vdrive = build_driving_vector(n, sigma, ports_in, eps, Vin=Vin)
        v = 1j * (G @ Vdrive)          # scaled node voltages
        V_phys = sigma * v             # unscaled/physical node voltages
        T = V_phys[b] / Vin            # <-- T = Vout / Vin
        Tabs.append(abs(T))
    return np.array(Tabs)

if __name__ == "__main__":
    n = 10
    connections = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9)]
    mode = "periodic"            # or "random"
    if mode == "periodic":
        Z_list = [50 if i%2==0 else 93 for i in range(len(connections))]
    else:
        import random
        Z_list = [random.choice([50,93]) for _ in range(len(connections))]
    ports_in  = {0: 50.0}
    ports_out = {9: 50.0}
    f = np.linspace(0, 600e6, 601)
    Tabs = T_voltage_ratio_vs_freq(f, connections, Z_list, ports_in, ports_out, n, Vin=1.0)
    plt.figure(figsize=(8,4))
    plt.plot(f/1e6, Tabs)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("|T| = |Vout / Vin| (node)")
    plt.title(f"|T| vs f   Z_list={Z_list}")
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
