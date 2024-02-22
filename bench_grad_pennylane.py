from pennylane import numpy as np
import networkx as nx
import time
import pandas as pd
from pennylane import qaoa
import pennylane as qml

np.random.seed(12345)

sizes = np.arange(6, 20, 2).tolist()

depths = [1, 2, 4, 6, 8, 16]

graphs = {
    (s, i): nx.random_regular_graph(3, s)
    for s in sizes
    for i in range(10)
}


data = []
for (s, i), g in graphs.items():
    print(s, i)
    dev = qml.device('lightning.qubit', wires=s)

    a = time.perf_counter()
    cost_h, mixer_h = qaoa.maxcut(g)
    def qaoa_layer(gamma, beta):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(beta, mixer_h)

    diag_time = time.perf_counter() - a

    for depth in depths:
        params = np.array(np.random.randn(2, depth), requires_grad=True)

        a = time.perf_counter()
        @qml.qnode(dev, diff_method="adjoint")
        def circuit(params):
            for w in range(s):
                qml.Hadamard(wirs=w)
            qml.layer(qaoa_layer, depth, params[1], params[0])
            return qml.expval(cost_h)

        grad = qml.grad(circuit)
        a = time.perf_counter()
        grad(params)
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

pd.DataFrame(data).to_csv("max_cut_grad_pennylane.csv")
