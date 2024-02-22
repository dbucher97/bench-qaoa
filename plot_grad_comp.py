import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

df_fq = pd.read_csv("max_cut_grad_fastqaoa.csv", index_col=0)
df_fq["sim"] = "fastqaoa"

df_pl = pd.read_csv("max_cut_grad_pennylane.csv", index_col=0)
df_pl["sim"] = "pennylane"

df = pd.concat([df_fq, df_pl], ignore_index=True)

df["total_time"] = df.sim_time + df.diag_time


hue_order = ["fastqaoa", "pennylane"]

#############################################################################
fig, ax = plt.subplots(figsize=(5, 4))

sns.set(style="whitegrid")
ax.set_title("gradient N = 18")


sns.lineplot(
    df.query("size == 18"),
    x="depth",
    y="sim_time",
    hue="sim",
    marker="o",
    hue_order=hue_order,
)
plt.grid()
plt.grid(which="minor", axis="y", alpha=0.1)

ax.set_ylabel("gradient eval time [s]")

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])

fig.tight_layout()
fig.savefig("plots/n18_grad_comp.png", dpi=150, bbox_inches="tight")

################################################################
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
ax.set_xticks(range(6, 20, 2))
ax.set_ylabel("gradient time [s]")

fig.tight_layout()

fig.savefig("plots/p6_gradient_comp.png", dpi=150, bbox_inches="tight")

