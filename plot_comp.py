import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

df_qk = pd.read_csv("../QOKit/max_cut_reg_qokit.csv", index_col=0)
df_qk["sim"] = "qokit"

df_fq = pd.read_csv("max_cut_reg_fastqaoa.csv", index_col=0)
df_fq["sim"] = "fastqaoa"

df_fq32 = pd.read_csv("max_cut_reg_fastqaoa_32.csv", index_col=0)
df_fq32["sim"] = "fastqaoa_32"
df_fq32["total_time"] = df_fq32.sim_time + df_fq32.diag_time


df_qi = pd.read_csv("max_cut_reg_qiskit.csv", index_col=0)
df_qi["sim"] = "qiskit"

df_pl = pd.read_csv("max_cut_reg_pennylane.csv", index_col=0)
df_pl["sim"] = "pennylane"

df = pd.concat([df_qk, df_qi, df_fq, df_pl], ignore_index=True)

df["total_time"] = df.sim_time + df.diag_time

hue_order = ["fastqaoa", "qokit", "qiskit", "pennylane"]

################################################################
fig, ax = plt.subplots(figsize=(5, 4))

sns.set(style="whitegrid")
ax.set_title("N = 24")

sns.lineplot(
    df.query("size == 24"),
    x="depth",
    y="sim_time",
    hue="sim",
    marker="o",
    hue_order=hue_order,
)
plt.grid()
plt.grid(which="minor", axis="y", alpha=0.1)

ax.set_ylabel("simulation time [s]")

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])

fig.tight_layout()
fig.savefig("plots/n24_comp.png", dpi=150, bbox_inches="tight")


################################################################
fig, ax = plt.subplots(figsize=(5, 4))

ax.set_title("depth = 6")

sns.set(style="whitegrid")
sns.lineplot(
    df.query("depth == 6"),
    x="size",
    y="sim_time",
    hue="sim",
    marker="o",
    hue_order=hue_order,
)
plt.grid(which="minor", axis="y", alpha=0.1)

ax.set_yscale("log")
ax.set_xticks(range(6, 26, 2))
ax.set_ylabel("simulation time [s]")

fig.tight_layout()

fig.savefig("plots/p6_comp.png", dpi=150, bbox_inches="tight")


#################################################################
fig, ax = plt.subplots(figsize=(5, 4))

ax.set_title("depth = 6")

sns.set(style="whitegrid")
sns.lineplot(
    df.query("depth == 6"),
    x="size",
    y="total_time",
    hue="sim",
    marker="o",
    hue_order=hue_order,
)
plt.grid(which="minor", axis="y", alpha=0.1)

ax.set_yscale("log")
ax.set_xticks(range(6, 26, 2))
ax.set_ylabel("total time [s]")

fig.tight_layout()

fig.savefig("plots/p6_comp_tot_time.png", dpi=150, bbox_inches="tight")
