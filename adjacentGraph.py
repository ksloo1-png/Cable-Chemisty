import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ---- Example: impedance adjacency matrix (0 = no edge) ----
# Use complex values if you like (e.g., 50+1j*2 means 50Ω + j2Ω)
Z = np.array([
    [0,   50,   0,    0],
    [50,   0,  93,    0],
    [0,   93,   0,   50],
    [0,   0,   50,    0],
], dtype=complex)

# ---- Build graph: undirected, edge attribute "Z" for impedance ----
G = nx.Graph()
n = Z.shape[0]
G.add_nodes_from(range(n))

# add edges where Z_ij != 0; store the complex impedance
for i in range(n):
    for j in range(i+1, n):
        Zij = Z[i, j]
        if Zij != 0:
            G.add_edge(i, j, Z=Zij)

# ---- Use positions (spring layout or your own coordinates) ----
pos = nx.spring_layout(G, seed=7)

# ---- Draw nodes/edges; color edges by |Z| and label with full Z ----
edge_Z = np.array([G[u][v]['Z'] for u, v in G.edges()])
edge_mag = np.abs(edge_Z)  # magnitude for coloring

# nodes
nx.draw_networkx_nodes(G, pos, node_size=600, edgecolors="k")
nx.draw_networkx_labels(G, pos, font_size=10)

# edges
ec = nx.draw_networkx_edges(
    G, pos, width=2.0, edge_color=edge_mag, edge_cmap=plt.cm.plasma
)

# edge labels: show real/imag neatly
def z_label(z):
    if isinstance(z, complex) and np.isclose(z.imag, 0):
        return f"{z.real:.1f} Ω"
    if isinstance(z, complex) and not np.isclose(z.imag, 0):
        sign = "+" if z.imag >= 0 else "−"
        return f"{z.real:.1f} {sign} j{abs(z.imag):.1f} Ω"
    return f"{z} Ω"

edge_labels = {(u, v): z_label(G[u][v]['Z']) for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

plt.title("Cable Network (edges colored by |Z|)")
plt.axis("off")
plt.tight_layout()
plt.show()
