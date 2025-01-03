import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap as tw
sns.set_theme(style="ticks")
import matplotlib.pyplot as plt

def optimal_bin(data):
    data = data.to_numpy()
    iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    h = 2 * iqr / len(data)**(1/3)
    return int((data.max() - data.min()) / h + 1)

if __name__=="__main__":
    path = sys.argv[1]
    prot_path = os.path.join(path, "proteome_data.csv")
    prot_df = pd.read_csv(prot_path, sep="\t", header=0, index_col=0)

    cor_path = os.path.join(os.path.join(path, "standard"), "cor_data.csv")
    cor_df = pd.read_csv(cor_path, sep="\t", header=0, index_col=0)

    joint_df = pd.DataFrame({"GC": prot_df["GC"],
                             "Codon_Cor": cor_df["codon_spear"],
                             "GC_Cor": cor_df["gc_spear"],
                             "Cor_Dif": np.abs(cor_df["codon_spear"]-cor_df["gc_spear"])})

    g = sns.JointGrid(data=joint_df, x="GC", y="Codon_Cor")
    g.plot_joint(sns.scatterplot, alpha=0.5, color="blue",
                 label="Codon correlation")
    sns.scatterplot(data=joint_df, x="GC", y="GC_Cor", alpha=0.5, color="red",
                    ax=g.ax_joint, label="GC correlation")

    sns.histplot(data=joint_df["GC"], ax=g.ax_marg_x, color="purple",
                 element="step")

    hist_y1, bins_y1 = np.histogram(joint_df["Codon_Cor"], density=True,
                                    bins=optimal_bin(joint_df["Codon_Cor"]))
    g.ax_marg_y.fill_betweenx(bins_y1[:-1], 0, hist_y1, step="pre", color="blue",
                              alpha=0.5)

    hist_y2, bins_y2 = np.histogram(joint_df["GC_Cor"], density=True,
                                    bins=optimal_bin(joint_df["GC_Cor"]))
    g.ax_marg_y.fill_betweenx(bins_y2[:-1], 0, hist_y2, step="pre", color="red",
                              alpha=0.5)

    g.ax_joint.set_xlabel("GC content")
    g.ax_joint.set_ylabel("Spearman correlation coefficient")
    title = tw.fill("Archaea - Spearman correlation coefficients "
                    "for genetic code: mold-protozoan-coelenterate mitochondrial and mycoplasma spiroplasma", 100)
    g.fig.suptitle(title)
    sns.move_legend(g.ax_joint, "upper left")
    g.fig.subplots_adjust(top=0.92)
    g.fig.set_figheight(10)
    g.fig.set_figwidth(15)

    plt.savefig("test.pdf", bbox_inches="tight")

    plt.close()
