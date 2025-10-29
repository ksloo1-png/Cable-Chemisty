import numpy as np

def build_Q(N, Z, periodic=False):
    """
    Construct the chiral block Q for a 1D chain with N nodes.
    Basis order is [white | black] where white = even nodes, black = odd nodes.

    Parameters
    ----------
    N : int
        Number of nodes (N >= 2 recommended).
    Z : float | complex | array-like
        Link impedances. If scalar: uniform impedance on all links.
        If array-like:
            - open chain: length N-1, Z[k] is impedance between k+1 and k+2 (1-based).
            - periodic:  length N,   Z[k] is impedance between k+1 and k+2 (wrap at N->1).
    periodic : bool
        If True, include the wrap link (N,1).

    Returns
    -------
    Q : np.ndarray (complex)
        Shape (N_white, N_black).
    Q_star : np.ndarray (complex)
        Conjugate-transpose of Q, shape (N_black, N_white).
    info : dict
        Useful metadata: {'white_nodes', 'black_nodes', 'sigma', 't_links', 'Z_links'}
    """
    if N < 1:
        raise ValueError("N must be >= 1")

    # Determine number of links and normalize Z to an array
    n_links = N if periodic else max(N - 1, 0)

    if np.isscalar(Z):
        Z_links = np.full(n_links, Z, dtype=complex)
    else:
        Z_arr = np.asarray(Z, dtype=complex)
        expected = n_links
        if Z_arr.size != expected:
            kind = "N" if periodic else "N-1"
            raise ValueError(f"Impedance list must have length {expected} for "
                             f"{'periodic' if periodic else 'open'} chain (got {Z_arr.size}).")
        Z_links = Z_arr.copy()

    # Basic checks
    if n_links > 0 and np.any(Z_links == 0):
        raise ValueError("Impedances must be non-zero.")

    # Node sets and index maps (1-based node labels; internal arrays are 0-based)
    black_nodes = [i for i in range(1, N + 1) if i % 2 == 1]  # odd
    white_nodes = [i for i in range(1, N + 1) if i % 2 == 0]  # even
    wb_index = {n: r for r, n in enumerate(white_nodes)}      # node -> row in Q
    bb_index = {n: c for c, n in enumerate(black_nodes)}      # node -> col in Q

    # Build edge list and map impedances to each edge
    # edges as tuples (i, j, Z_ij) with i<->j neighbors (1-based labels)
    edges = []
    for k in range(1, N):  # 1..N-1
        edges.append((k, k + 1, Z_links[k - 1] if n_links > 0 else None))
    if periodic and N >= 2:
        edges.append((N, 1, Z_links[-1]))

    # Compute sigma_n = sqrt(sum_{adjacent to n} 1/Z)
    sigma = np.zeros(N, dtype=complex)  # 0-based
    invZ_sums = np.zeros(N, dtype=complex)
    for (i, j, Zij) in edges:
        inv = 1.0 / Zij
        invZ_sums[i - 1] += inv
        invZ_sums[j - 1] += inv
    sigma = np.sqrt(invZ_sums)

    # Couplings t_{ij} = sigma_i * sigma_j / Z_{ij} for each edge
    t_by_edge = []  # list of (i,j,t_ij)
    for (i, j, Zij) in edges:
        ti = sigma[i - 1] * sigma[j - 1] / Zij
        t_by_edge.append((i, j, ti))

    # Allocate Q (rows=white, cols=black)
    Q = np.zeros((len(white_nodes), len(black_nodes)), dtype=complex)

    # Fill Q: for each edge, place t into the (white_row, black_col) cell
    for (i, j, tij) in t_by_edge:
        # Determine which endpoint is white and which is black
        if i in white_nodes and j in black_nodes:
            Q[wb_index[i], bb_index[j]] += tij
        elif j in white_nodes and i in black_nodes:
            Q[wb_index[j], bb_index[i]] += tij
        else:
            # For a bipartite chain, every edge connects a white to a black; this is just a guard
            raise RuntimeError(f"Edge ({i},{j}) does not connect white<->black with current coloring.")

    Q_star = Q.conj().T

    info = {
        "white_nodes": white_nodes,
        "black_nodes": black_nodes,
        "sigma": sigma,
        "t_links": t_by_edge,
        "Z_links": Z_links,
        "edges": edges,
    }
    return Q, Q_star, info


# -----------------------
if __name__ == "__main__":
    # Example 1: N=4, open chain, Z = [50, 97, 50]
    N = 4
    Z_list = [50, 97, 50]
    Q, Q_star, info = build_Q(N, Z_list, periodic=False)
    print(f"N={N}, white={info['white_nodes']}, black={info['black_nodes']}")
    print("Q shape:", Q.shape)
    print("Q =\n", Q)
    print("Q* =\n", Q_star)

    # Example 2: N=5, open chain, Z = [50, 97, 50, 97]
    N = 5
    Z_list = [50, 97, 50, 97]
    Q, Q_star, info = build_Q(N, Z_list, periodic=False)
    print(f"\nN={N}, white={info['white_nodes']}, black={info['black_nodes']}")
    print("Q shape:", Q.shape)
    print("Q =\n", Q)
    print("Q* =\n", Q_star)

    # Optional: periodic uniform example
    # Qp, Qp_star, _ = build_Q(6, 75, periodic=True)
    # print("\nPeriodic N=6, uniform Z=75Ω")
    # print("Q (periodic) =\n", Qp)
