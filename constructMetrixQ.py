import numpy as np

# ---------------- Core (unchanged) ----------------
def _edges_from_Zlist(N, Z_list):
    Z = np.asarray(Z_list, dtype=complex)
    if Z.size == N - 1:
        periodic = False  # always open chain
    else:
        raise ValueError(f"Z_list must have length N-1 for open chain (got {Z.size}).")
    if np.any(Z == 0):
        raise ValueError("Impedances must be non-zero.")
    edges = [(k, k + 1, Z[k - 1]) for k in range(1, N)]  # (1,2)...(N-1,N)
    return edges, Z, periodic

def build_Q(N, Z_list):
    """Construct Q for open chain (basis [white | black])."""
    if N < 1:
        raise ValueError("N must be >= 1")
    edges, Z_links, periodic = _edges_from_Zlist(N, Z_list)

    black = [i for i in range(1, N + 1) if i % 2 == 1]
    white = [i for i in range(1, N + 1) if i % 2 == 0]
    wb_idx = {n: r for r, n in enumerate(white)}
    bb_idx = {n: c for c, n in enumerate(black)}

    invZ_sum = np.zeros(N, dtype=complex)
    for (i, j, Zij) in edges:
        inv = 1.0 / Zij
        invZ_sum[i - 1] += inv
        invZ_sum[j - 1] += inv
    sigma = np.sqrt(invZ_sum)

    Q = np.zeros((len(white), len(black)), dtype=complex)
    for (i, j, Zij) in edges:
        tij = sigma[i - 1] * sigma[j - 1] / Zij
        if i in white and j in black:
            Q[wb_idx[i], bb_idx[j]] += tij
        elif j in white and i in black:
            Q[wb_idx[j], bb_idx[i]] += tij
    return Q, sigma, {"white": white, "black": black, "Z_links": Z_links}

def analyze_chain(N, Z_list):
    """Return epsilon^2 and epsilon for given Z_list (open chain)."""
    Q, sigma, info = build_Q(N, Z_list)
    QQs = Q @ Q.conj().T
    evals, _ = np.linalg.eigh(QQs)
    evals = np.sort(evals)[::-1]
    eps = np.sqrt(np.clip(evals, 0, None))
    return {"epsilon2": evals, "epsilon": eps, "Q": Q, "QQ*": QQs, "sigma": sigma, **info}

# ---------------- Z_list helpers (only 50 or 93) ----------------
def z_periodic_open(N, a=50, b=93, start_with='a'):
    """Alternating [a,b,a,b,...] impedances for open chain."""
    length = N - 1
    pattern = np.array([a, b], dtype=float)
    if start_with == 'b':
        pattern = pattern[::-1]
    return np.resize(pattern, length)

def z_random_open(N, a=50, b=93, seed=None):
    """Random sequence of {a,b} impedances for open chain."""
    rng = np.random.default_rng(seed)
    return rng.choice([a, b], size=N - 1)

# ---------------- Example usage ----------------
if __name__ == "__main__":
    N = 10

    # Alternating (periodic pattern)
    Z_alt = z_periodic_open(N, 50, 93)
    res_alt = analyze_chain(N, Z_alt)
    print("Alternating 50/93 open chain:")
    print("Z_list =", Z_alt.tolist())
    print("epsilon =", np.round(res_alt["epsilon"], 6))

    # Random 50/93 open chain
    Z_rand = z_random_open(N, 50, 93, seed=42)
    res_rand = analyze_chain(N, Z_rand)
    print("\nRandom 50/93 open chain:")
    print("Z_list =", Z_rand.tolist())
    print("epsilon =", np.round(res_rand["epsilon"], 6))

