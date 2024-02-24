import numpy as np
import networkx as nx
import time
import pandas as pd
import qokit

np.random.seed(12345)

sizes = np.arange(6, 26, 2)

depths = [1, 2, 4, 6, 8, 16]

graphs = {(s, i): nx.random_regular_graph(3, s) for s in sizes for i in range(10)}


simclass = qokit.fur.choose_simulator(name="c")
print(simclass)


data = []
for (s, i), g in graphs.items():
    print(s, i)
    terms = [(-1, e) for e in g.edges()]

    a = time.perf_counter()
    sim = simclass(s, terms=terms)
    diag_time = time.perf_counter() - a

    for depth in depths:
        gamma, beta = np.random.randn(2, depth)

        a = time.perf_counter()
        sim.simulate_qaoa(gamma, beta)
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

pd.DataFrame(data).to_csv("max_cut_reg_qokit.csv")
