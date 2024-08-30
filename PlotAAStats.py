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


def optimal_bin(data):
    data = data.to_numpy()
    iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    h = 2 * iqr / len(data)**(1/3)
    return int((data.max() - data.min()) / h + 1)


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

        # Plot the distributions of average amino acid counts, GC contents and
        # protein length of each proteome
        if(code_name == "Standard"):
            # Plot amino acid distribution
            amino_acids = list(data_df.columns[:20])
            amino_df = data_df[amino_acids]
            # Remove outliers under or above the IRQ-limit
            q1 = amino_df.quantile(0.25).to_numpy()
            q3 = amino_df.quantile(0.75).to_numpy()
            limit_bot = q1 - 1.5 * (q3 - q1)
            limit_top = q3 + 1.5 * (q3 - q1)
            aa_data_filtered = []
            for i,aa in enumerate(amino_acids):
                aa_data_filtered.append(data_df[aa][(data_df[aa]>=limit_bot[i])
                                        & (data_df[aa]<=limit_top[i])])

            plt.boxplot(aa_data_filtered, labels=amino_acids, vert=False,
                        notch=True, patch_artist=True,
                        flierprops={"markerfacecolor": "red", "alpha": 0.5})
            title = tw.fill(f"{name} - Average amino acid frequency", 100)
            plt.title(title)
            plt.xlabel("Average amino acid frequency")
            plt.ylabel("Amino acid")
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{path}/aa_frequency.{ext}", bbox_inches="tight")

            plt.close()

            # Plot GC content distribution
            bins = optimal_bin(data_df["GC"])
            sns.histplot(data_df, x="GC", bins=bins, alpha=0.4, color="maroon",
                         kde=True, line_kws={"linewidth": 2, "linestyle": "--"})
            title = tw.fill(f"{name} - Density of average proteome GC content",
                            100)
            plt.title(title)
            plt.xlabel("Average GC content")
            plt.ylabel("GC frequency")
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{path}/gc_frequency.{ext}", bbox_inches="tight")

            plt.close()

            # Plot protein length distribution
            bins = optimal_bin(data_df["Length"])
            sns.histplot(data_df, x="Length", bins=bins, alpha=0.4,
                         color="maroon", kde=True,
                         line_kws={"linewidth": 2, "linestyle": "--"})
            title = tw.fill(f"{name} - Density of average proteome length", 100)
            plt.title(title)
            plt.xlabel("Average protein length")
            plt.ylabel("Length frequency")
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{path}/length_frequency.{ext}",
                            bbox_inches="tight")

            plt.close()

        # Take the median of each value over all proteomes
        med_data_df = data_df.median(axis=0)

        fig, axes = plt.subplots(2, 2)
        i = 0
        j = 0
        for type,aa_list in aa_groups.items():
            x_data = np.arange(len(aa_list))
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

        fig, axes = plt.subplots(2, 2)
        i = 0
        j = 0
        data_name = f"Data type: {len(data_df)}"
        for sig in [1, 0.05]:
            for cor_type in ["pearson", "spearman"]:
                cor_df = pd.DataFrame()
                ridge_df = pd.DataFrame()
                for calc_type in ["codon", "gc", "cost", "cost_codon",
                                  "cost_gc"]:
                    cor_col = f"{cor_type} {calc_type}"
                    p_col = f"p-{cor_type} {calc_type}"
                    label = calc_type.capitalize().replace("_", "+")
                    if(calc_type=="gc"):
                        label = "Codon+GC"
                    elif(calc_type=="cost_gc"):
                        label = "Cost+codon+GC"

                    data = data_df[[cor_col, p_col]]
                    data = data_df[data_df[p_col] <= sig]
                    if(sig == 0.05):
                        label += f": {len(data)}"

                    df = pd.DataFrame(index=data.index, columns=[data_name,
                                                                 "Value"])
                    df[data_name] = [label] * len(data)
                    df["Value"] = data[cor_col]
                    cor_df = pd.concat([cor_df, df])

                axes[i, j]=sns.kdeplot(ax=axes[i,j], data=cor_df, x="Value",
                                       hue=data_name, fill=True, alpha=0.5,
                                       common_norm=False, linewidth=2.5)

                sns.move_legend(axes[i,j], "upper left")
                axes[i,j].set_xlabel(None)
                axes[i,j].set_ylabel("Density")
                title = f" {cor_type.capitalize()} correlation coefficients"
                if(sig == 1):
                    title = "All" + title
                else:
                    title = "Significant" + title

                axes[i,j].set_title(title)

                j = 1 if i == 1 else j
                i = 0 if i == 1 else i + 1

        x_min = min(min([ax.get_xlim() for ax in axes.reshape(-1)]))
        x_max = max(max([ax.get_xlim() for ax in axes.reshape(-1)]))
        for ax in axes.reshape(-1):
            ax.set_xlim(x_min, x_max)

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
