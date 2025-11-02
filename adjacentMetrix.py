import numpy as np

def build_Q(N, Z, periodic=False):
    """Construct Q for an N-node 1D chain (basis order [white | black])."""
    if N < 1:
        raise ValueError("N must be >= 1")

    n_links = N if periodic else max(N - 1, 0)

    # Normalize impedance input
    if np.isscalar(Z):
        Z_links = np.full(n_links, Z, dtype=complex)
    else:
        Z_arr = np.asarray(Z, dtype=complex)
        expected = n_links
        if Z_arr.size != expected:
            raise ValueError(f"Impedance list must have length {expected} "
                             f"for {'periodic' if periodic else 'open'} chain (got {Z_arr.size}).")
        Z_links = Z_arr.copy()

    if n_links > 0 and np.any(Z_links == 0):
        raise ValueError("Impedances must be non-zero.")

    # Define sublattices
    black_nodes = [i for i in range(1, N + 1) if i % 2 == 1]  # odd
    white_nodes = [i for i in range(1, N + 1) if i % 2 == 0]  # even
    wb_index = {n: r for r, n in enumerate(white_nodes)}
    bb_index = {n: c for c, n in enumerate(black_nodes)}

    # Build edge list
    edges = [(k, k + 1, Z_links[k - 1]) for k in range(1, N)]
    if periodic and N >= 2:
        edges.append((N, 1, Z_links[-1]))

    # Compute sigma_n = sqrt(sum 1/Z for adjacent links)
    invZ_sums = np.zeros(N, dtype=complex)
    for (i, j, Zij) in edges:
        inv = 1.0 / Zij
        invZ_sums[i - 1] += inv
        invZ_sums[j - 1] += inv
    sigma = np.sqrt(invZ_sums)

    # Coupling t_ij = sigma_i * sigma_j / Z_ij
    t_by_edge = [(i, j, sigma[i - 1] * sigma[j - 1] / Zij) for (i, j, Zij) in edges]

    # Build Q (rows=white, cols=black)
    Q = np.zeros((len(white_nodes), len(black_nodes)), dtype=complex)
    for (i, j, tij) in t_by_edge:
        if i in white_nodes and j in black_nodes:
            Q[wb_index[i], bb_index[j]] += tij
        elif j in white_nodes and i in black_nodes:
            Q[wb_index[j], bb_index[i]] += tij
        else:
            raise RuntimeError(f"Edge ({i},{j}) does not connect white↔black")

    Q_star = Q.conj().T
    return Q, Q_star, white_nodes, black_nodes, sigma


def analyze_chain(N, Z, periodic=False):
    """Compute Q, Q*, Q Q*, eigenvalues epsilon^2 and epsilon."""
    Q, Q_star, whites, blacks, sigma = build_Q(N, Z, periodic)
    QQs = Q @ Q.conj().T  # Hermitian
    evals, evecs = np.linalg.eigh(QQs)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    eps = np.sqrt(np.clip(evals, 0, None))
    print(f"\n=== Chain analysis: N={N} ({'periodic' if periodic else 'open'}) ===")
    print(f"white nodes: {whites}, black nodes: {blacks}")
    print(f"Q shape: {Q.shape}")
    print("Eigenvalues of Q Q* (epsilon^2):", evals)
    print("Corresponding epsilon:", eps)
    return {"Q": Q, "Q*": Q_star, "QQ*": QQs, "epsilon2": evals, "epsilon": eps,
            "white": whites, "black": blacks, "sigma": sigma}


# -------------------------------
# Example usage
# -------------------------------

# Example 1: N=4, open chain
res4 = analyze_chain(4, [50, 97, 50])

# Example 2: N=5, open chain
res5 = analyze_chain(5, [50, 97, 50, 97])

# Example 3 (optional): periodic uniform chain
# res6 = analyze_chain(6, 75, periodic=True)
