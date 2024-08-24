import os
import sys
import numpy as np
import sympy as sp
import pandas as pd
import textwrap as tw
import seaborn as sns
import collections as col
import scipy.stats as sci
import equation_functions as ef
from Bio.Data import CodonTable
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# Function for sorting and returning the amino acids based on a genetic code
def get_amino_acids(id):
    codon_table = CodonTable.unambiguous_dna_by_id[id]
    codon_map = col.Counter(codon_table.forward_table.values())
    codon_map = dict(sorted(codon_map.items(), key=lambda x: (x[1], x[0])))
    return list(codon_map.keys())


# Function to calculate the sum of values
def count_amino_acids(codon_dis, canon_codons):
    return sum(
        int(codon.split(":")[1])
        for codon in codon_dis.split(";")
        if codon.split(":")[0] in canon_codons
    )


# Function to create ridge plot of correlations over all genetic codes
def plot_ridge(data, name, type, output):
    ridge_output = os.path.join(output, type.lower())
    os.makedirs(ridge_output, exist_ok=True)

    # Split data into 4 groups
    categories = list(data.keys())
    groups = [categories[i::4] for i in range(4)]

    all_data_df = pd.DataFrame()
    for i, group in enumerate(groups):
        df_list = []
        for category in group:
            for factor, values in data[category].items():
                df_list.append(pd.DataFrame({
                    f"{type} correlation coefficient": category,
                    "Factor": factor,
                    "Value": values
                }))

        df = pd.concat(df_list)
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        # Initialize the FacetGrid object
        g = sns.FacetGrid(df, row=f"{type} correlation coefficient",
                          hue="Factor", aspect=15, height=0.5,
                          palette=kde_col_pal)

        # Draw the densities
        g.map(sns.kdeplot, "Value", bw_adjust=0.5, clip_on=False, fill=True,
              alpha=0.5, linewidth=2.5)
        g.map(sns.kdeplot, "Value", clip_on=False, color="white", lw=2,
              bw_adjust=0.5)
        g.refline(y=0, linewidth=2, linestyle="-", color="black", clip_on=False)

        # Label function
        def label(x, color, label):
            label = tw.fill(x.unique()[0], 30)
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color="black",
                    ha="left", va="center", transform=ax.transAxes,
                    fontsize=10)

        g.map(label, f"{type} correlation coefficient")

        # Overlap subplots
        g.figure.subplots_adjust(hspace=-.25)

        # Remove axes details
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        # Add a legend
        handles = [plt.Line2D([0], [0], color=color, lw=4)
                   for color in kde_col_pal]
        labels = df["Factor"].unique()
        legend = g.fig.legend(handles, labels, bbox_to_anchor=(0.3, 0.9),
                              ncol=1, fancybox=True, fontsize=12)
        legend.get_frame().set_facecolor("white")

        title = f"{name} - {type} correlations over all genetic codes"
        g.fig.suptitle(title, y=0.9)
        g.fig.set_figheight(10)
        g.fig.set_figwidth(15)
        for ext in ["svg", "pdf"]:
            plt.savefig(f"{ridge_output}/cor_ridge_plot_{i+1}.{ext}",
                        bbox_inches="tight")

        plt.close()

        all_data_df = pd.concat([all_data_df, df])

    all_data_df = all_data_df.rename(columns={f"{type} correlation coefficient":
                                              "Code"})
    all_data_df.to_csv(f"{ridge_output}/cor_data.csv", sep="\t", index=False)


if __name__ == "__main__":
    path, name = sys.argv[1:3]

    cmap = plt.get_cmap("viridis")

    folders = [os.path.join(path, folder)
               for folder in os.listdir(path)
               if os.path.isdir(os.path.join(path, folder))]

    # Group amino acids based on their attributes
    aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
                 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
                 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}

    for folder in folders:
        code_name = os.path.basename(folder).capitalize().replace("_", " ")
        print(f"Current code: {code_name}...")

        data_df = pd.read_csv(os.path.join(folder, "proteome_freqs.csv"),
                              sep="\t", index_col=0)
        # Take the median of each value over all proteomes
        med_data_df = data_df.median(axis=0)

        fig, axes = plt.subplots(2, 2)
        i = 0
        j = 0
        for type,aa_list in aa_groups.items():
            x_data = np.arange(len(aa_list))
            # -0.25 -0.15 -0.05 0.05 0.15 0.25
            wid = 0.1
            b_pos = -0.25
            col = 0.15
            hatch = "//"
            # Plot observed median frequency of amino acids
            obs_aas = np.array([med_data_df[aa] for aa in aa_list])
            axes[i,j].bar(x_data+b_pos, obs_aas, width=wid, color=cmap(col),
                          edgecolor="black", linewidth=0.75, hatch=hatch,
                          label="Observed", zorder=2)

            c_pos = 0.2
            for calc_type in ["codon", "gc", "cost", "cost_codon", "cost_gc"]:
                hatch = "\\\\" if hatch=="//" else "//"
                b_pos += wid
                col += 0.15
                # Plot median frequency of amino acids of the current type
                obs_aas = np.array([med_data_df[aa+f"_{calc_type}"]
                                    for aa in aa_list])
                label = calc_type.capitalize().replace("_", "+")
                if(calc_type=="gc"):
                    label = "Codon+GC"
                elif(calc_type=="cost_gc"):
                    label = "Cost+codon+GC"

                axes[i,j].bar(x_data+b_pos, obs_aas, width=wid, color=cmap(col),
                              edgecolor="black", linewidth=0.75, hatch=hatch,
                              zorder=2, label=label)

                axes[i,j].grid(visible=True, which="major", color="#999999",
                               linestyle="dotted", alpha=0.5, zorder=0)

                axes[i,j].set_xticks(np.arange(len(aa_list)), aa_list)
                axes[i,j].set_xlabel("Amino acid")
                axes[i,j].set_title(f"{type} amino acids")
                if(j == 0):
                    axes[i,j].set_ylabel("Mean amino acid frequency")
                    if(i == 0):
                        pear_ar = [f"pearson {calc_type}",
                                   f"p-pearson {calc_type}"]
                        pcc,pcc_p = med_data_df[pear_ar]
                        spear_ar = [f"spearman {calc_type}",
                                    f"p-spearman {calc_type}"]
                        r2, r2_p = med_data_df[spear_ar]
                        label = tw.fill(f"${label}$ correlation:", 25)
                        axes[i,j].text(1.03, c_pos, f"{label}\n"
                            f"  - $pcc$: {pcc:.2f}; $p_{{pcc}}$: {pcc_p:.1e}\n"
                            f"  - $r2$: {r2:.2f}; $p_{{r2}}$: {r2_p:.1e}",
                            transform=axes[0,0].transAxes, fontsize=11,
                            verticalalignment="top", bbox=dict(boxstyle="round",
                                facecolor="white", edgecolor="grey", alpha=0.5))
                c_pos -= 0.25

            j = 1 if i == 1 else j
            i = 0 if i == 1 else i + 1

        y_max = max(max([ax.get_ylim() for ax in axes.reshape(-1)]))
        for ax in axes.reshape(-1):
            ax.set_ylim(0, y_max)

        axes[0,0].legend(loc="upper center", bbox_to_anchor=(1.23, 0.8),
                         fancybox=True, fontsize=12)

        fig.subplots_adjust(wspace=0.6, hspace=0.3)
        title = tw.fill(f"{name} - Correlation of amino acid frequency "
                        f"for genetic code: {code_name}", 100)
        fig.suptitle(title, fontsize=15, y=0.95)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        for ext in ["svg", "pdf"]:
            plt.savefig(f"{folder}/cor_bar_plot.{ext}", bbox_inches="tight")

        plt.close()

        fig, axes = plt.subplots(2)
        i = 0
        for cor_type in ["pearson", "spearman"]:
            cor_df = pd.DataFrame()
            ridge_df = pd.DataFrame()
            for calc_type in ["codon", "gc", "cost", "cost_codon", "cost_gc"]:
                label = calc_type.capitalize().replace("_", "+")
                if(calc_type=="gc"):
                    label = "Codon+GC"
                elif(calc_type=="cost_gc"):
                    label = "Cost+codon+GC"

                data = data_df[f"{cor_type} {calc_type}"]
                df = pd.DataFrame(index=data.index, columns=["Data type",
                                                             "Value"])
                df["Data type"] = [label] * len(data)
                df["Value"] = data
                cor_df = pd.concat([cor_df, df])



            axes[i]=sns.kdeplot(ax=axes[i], data=cor_df, x="Value",
                                hue="Data type", fill=True, common_norm=False,
                                alpha=0.5, linewidth=2.5)

            sns.move_legend(axes[i], "upper left")
            axes[i].set_xlabel(None)
            axes[i].set_ylabel("Density")
            title = f"{cor_type.capitalize()} correlation coefficient"
            axes[i].set_title(title)
            i += 1

        y_max = max(max([ax.get_ylim() for ax in axes.reshape(-1)]))
        for ax in axes.reshape(-1):
            ax.set_ylim(0, y_max)

        fig.subplots_adjust(hspace=0.3)
        title = tw.fill(f"{name} - Density of correlation coefficients "
                        f"for genetic code: {code_name}", 100)
        fig.suptitle(title, fontsize=15, y=0.95)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        for ext in ["svg", "pdf"]:
            plt.savefig(f"{folder}/cor_density_plot.{ext}", bbox_inches="tight")

        plt.close()
