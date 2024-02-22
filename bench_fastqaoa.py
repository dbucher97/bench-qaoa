import numpy as np
import networkx as nx
import time
import pandas as pd
from fastqaoa import Diagonals, qaoa

np.random.seed(12345)

sizes = np.arange(6, 26, 2)

depths = [1, 2, 4, 6, 8, 16]

graphs = {
    (s, i): nx.random_regular_graph(3, s)
    for s in sizes
    for i in range(10)
}


data = []
for (s, i), g in graphs.items():
    print(s, i)
    terms = {e: 2 for e in g.edges()}
    terms.update({(v,): -g.degree(v) for v in g.nodes()})

    a = time.perf_counter()
    dg = Diagonals.brute_force_hamiltonian(s, terms)
    diag_time = time.perf_counter() - a

    for depth in depths:
        gamma, beta = np.random.randn(2, depth)

        a = time.perf_counter()
        qaoa(dg, gamma, beta)
        sim_time = time.perf_counter() - a

        data.append(
            {
                "size": s,
                "instance": i,
                "depth": depth,
                "diag_time": diag_time,
                "sim_time": sim_time,
            }
        )

pd.DataFrame(data).to_csv("max_cut_reg_fastqaoa_32.csv")
