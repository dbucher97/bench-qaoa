import numpy as np
import networkx as nx
import time
import pandas as pd
from pennylane import qaoa
import pennylane as qml

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
    dev = qml.device('lightning.qubit', wires=s)


    a = time.perf_counter()
    cost_h, mixer_h = qaoa.maxcut(g)
    def qaoa_layer(gamma, beta):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(beta, mixer_h)
    print(s, i)


    diag_time = time.perf_counter() - a

    for depth in depths:
        gamma, beta = np.random.randn(2, depth)

        a = time.perf_counter()
        @qml.qnode(dev)
        def circuit(betas, gammas):
            for w in range(s):
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, depth, betas, gammas)
            return qml.state()

        circuit(beta, gamma)
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

pd.DataFrame(data).to_csv("max_cut_reg_pennylane.csv")
