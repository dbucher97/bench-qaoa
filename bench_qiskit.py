import numpy as np
import networkx as nx
import time
import pandas as pd
from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import Aer

sim = Aer.get_backend("aer_simulator")
sim.set_options(max_parallel_threads=1)

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
    qr = QuantumRegister(s)
    qci = QuantumCircuit(qr)
    beta_p = Parameter("beta")
    gamma_p = Parameter("gamma")

    a = time.perf_counter()

    for u, v in g.edges():
        qci.crz(-2 * gamma_p, qr[u], qr[v])
    qci.rx(-2 * beta_p, qr)
    qci = transpile(qci, sim)


    diag_time_i = time.perf_counter() - a

    for depth in depths:
        gammas, betas = np.random.randn(2, depth)

        qc = QuantumCircuit(qr)

        a = time.perf_counter()
        qc.h(qr)
        for beta, gamma in zip(betas, gammas):
            qc.append(qci.assign_parameters({beta_p: beta, gamma_p: gamma}), qr)
        qc.save_statevector()
        qc = transpile(qc, sim)
        diag_time = diag_time_i + time.perf_counter() - a
        a = time.perf_counter()
        res = sim.run(qc).result()
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

pd.DataFrame(data).to_csv("max_cut_reg_qiskit.csv")
