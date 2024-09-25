import os
import sys
import numpy as np
import pandas as pd
import textwrap as tw
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


def optimal_bin(data):
    data = data.to_numpy()
    iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    h = 2 * iqr / len(data)**(1/3)
    return int((data.max() - data.min()) / h + 1)


if __name__ == "__main__":
    path,name = sys.argv[1:3]

    cmap = plt.get_cmap("viridis")

    folders = [os.path.join(path, folder)
               for folder in os.listdir(path)
               if os.path.isdir(os.path.join(path, folder))]

    # Get the canonical amino acid order
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]
    # Group amino acids based on their attributes
    aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
                 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
                 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}
    aas_cod = [f"{aa}_cod" for aa in amino_acids]
    aas_gc = [f"{aa}_gc" for aa in amino_acids]

    for folder in folders:
        # Name of the genetic code
        genetic_name = os.path.basename(folder).split(".")[0]
        code_name = genetic_name.capitalize().replace("_", " ")
        print(f"Current code: {code_name}...")

        # Proteome data
        data_df = pd.read_csv(os.path.join(folder, "proteome_cor_data.csv"),
                              sep="\t", index_col=0)

        # Plot the distributions of mean amino acid counts, GC contents and
        # mean proteome length
        if(code_name == "Standard"):
            # Plot amino acid distribution
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
            title = tw.fill(f"{name} - Mean amino acid frequency per proteome",
                            100)
            plt.title(title)
            plt.xlabel("Amino acid frequency")
            plt.ylabel("Amino acid")
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{path}/aa_frequency.{ext}", bbox_inches="tight")

            plt.close()

            # Plot GC content distribution
            bins = optimal_bin(data_df["GC"])
            sns.histplot(data_df, x="GC", bins=bins, alpha=0.4, color="maroon",
                         kde=True, line_kws={"linewidth": 2, "linestyle": "--"})
            title = tw.fill(f"{name} - Density of mean proteome GC content",
                            100)
            plt.title(title)
            plt.xlabel("GC content")
            plt.ylabel("GC density")
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{path}/gc_frequency.{ext}", bbox_inches="tight")

            plt.close()

            # Plot mean proteome length distribution
            bins = optimal_bin(data_df["Length"])
            sns.histplot(data_df, x="Length", bins=bins, alpha=0.4,
                         color="maroon", kde=True, log_scale=10,
                         line_kws={"linewidth": 2, "linestyle": "--"})
            title = tw.fill(f"{name} - Density of mean proteome length", 100)
            plt.title(title)
            plt.xlabel("log10-Proteome length")
            plt.ylabel("Length density")
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{path}/length_frequency.{ext}",
                            bbox_inches="tight")

            plt.close()

        cor_types = {"cod": ["Codon", data_df[aas_cod]],
                     "gc": ["Codon+GC", data_df[aas_gc]]}

        fig, axes = plt.subplots(2, 2)
        i = 0
        j = 0
        for type,aa_list in aa_groups.items():
            x_data = np.arange(len(aa_list))
            wid = 0.25
            b_pos = -0.25
            col = 0.3
            # Plot observed mean relative abundance and standard deviation of
            # each amino acid
            list_data = list(data_df[aa_list].mean(axis=0))
            list_data_err = list(data_df[aa_list].std(axis=0))
            axes[i,j].bar(x_data+b_pos, list_data, yerr=list_data_err,
                          width=wid, color=cmap(col), edgecolor="black",
                          linewidth=0.75, label="Observed", capsize=3, zorder=2)

            c_pos = 0.65
            for cor_type,data in cor_types.items():
                cols = [f"{aa}_{cor_type}" for aa in aa_list]
                list_data = list(data[1][cols].mean(axis=0))
                list_data_err = list(data[1][cols].std(axis=0))
                b_pos += wid
                col += 0.3
                label = data[0]
                axes[i,j].bar(x_data+b_pos, list_data, yerr=list_data_err,
                              width=wid, color=cmap(col), edgecolor="black",
                              label=label, linewidth=0.75, capsize=3, zorder=2)

                axes[i,j].grid(visible=True, which="major", color="#999999",
                               linestyle="dotted", alpha=0.5, zorder=0)

                axes[i,j].set_xticks(np.arange(len(aa_list)), aa_list)
                axes[i,j].set_xlabel("Amino acid")
                axes[i,j].set_title(f"{type} amino acids")
                if(j == 0):
                    axes[i,j].set_ylabel("Mean amino acid frequency")
                    if(i == 0):
                        pear_ar = [f"pearson_{cor_type}",
                                   f"p_pearson_{cor_type}"]
                        pcc,pcc_p = data_df[pear_ar].mean(axis=0)

                        spear_ar = [f"spearman_{cor_type}",
                                    f"p_spearman_{cor_type}"]
                        r2, r2_p = data_df[spear_ar].mean(axis=0)
                        label = tw.fill(f"${label}$ correlation:", 30)

                        axes[i,j].text(1.03, c_pos, f"{label}\n"
                            f"  - Mean Pearson:\n"
                            f"    - Coefficient: {pcc:.5f}\n"
                            f"    - p-value: {pcc_p:.5e}\n"
                            f"\n  - Mean Spearman:\n"
                            f"    - Coefficient: {r2:.5f}\n"
                            f"    - p-value: {r2_p:.5e}",
                            transform=axes[0,0].transAxes, fontsize=11,
                            verticalalignment="top", linespacing=1.5,
                            bbox=dict(boxstyle="round", facecolor="white",
                                      edgecolor="grey", alpha=0.5))
                c_pos -= 0.6

            j = 1 if i == 1 else j
            i = 0 if i == 1 else i + 1

        y_max = max(max([ax.get_ylim() for ax in axes.reshape(-1)]))
        for ax in axes.reshape(-1):
            ax.set_ylim(0, y_max)

        axes[0,0].legend(loc="upper center", bbox_to_anchor=(1.19, 1.025),
                         fancybox=True, fontsize=12)

        fig.subplots_adjust(wspace=0.6, hspace=0.3)
        title = tw.fill(f"{name} - Mean proteome amino acid frequency "
                        f"for genetic code: {code_name}", 100)
        fig.suptitle(title, fontsize=15, y=0.95)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        for ext in ["svg", "pdf"]:
            plt.savefig(f"{folder}/cor_bar_plot.{ext}", bbox_inches="tight")

        plt.close()

        fig, axes = plt.subplots(2)
        data_name = "Correlation type:"
        for i,cor in enumerate(["pearson", "spearman"]):
            cor_df = pd.DataFrame()
            for cor_type in cor_types:
                cor_col,p_col = [f"{cor}_{cor_type}", f"p_{cor}_{cor_type}"]
                label = cor_types[cor_type][0]
                data = data_df[[cor_col, p_col]]
                df = pd.DataFrame(index=data.index, columns=[data_name,
                                                             "Value"])
                df[data_name] = [label] * len(data)
                df["Value"] = data[cor_col]
                cor_df = pd.concat([cor_df, df])

            sns.kdeplot(ax=axes[i], data=cor_df, x="Value", hue=data_name,
                        fill=True, alpha=0.5, common_norm=False, linewidth=2.5)

            sns.move_legend(axes[i], "upper left")
            axes[i].set_xlabel(None)
            axes[i].set_ylabel("Density")
            title = f"{cor.capitalize()} correlation coefficients"
            axes[i].set_title(title)

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

        mean_amino_acids = data_df[amino_acids].mean(axis=0)
        reg_df = pd.DataFrame({"Amino acid frequency": mean_amino_acids,
                               "Codon": [data_df[f"{aa}_cod"].mean()
                                         for aa in amino_acids],
                               "Codon+GC": [data_df[f"{aa}_gc"].mean()
                                            for aa in amino_acids],
                               "Codon+Cost": [data_df[f"{aa}_ener_cod"].mean()
                                              for aa in amino_acids],
                               "Codon+GC+Cost": [data_df[f"{aa}_ener_gc"].mean()
                                                 for aa in amino_acids]})

        fig, axes = plt.subplots(2, 2)
        i = 0
        j = 0
        for type in ["Codon", "Codon+GC", "Codon+Cost", "Codon+GC+Cost"]:
            sns.regplot(data=reg_df, x=f"{type}", y="Amino acid frequency",
                        line_kws={"color": "orange"}, scatter_kws={"s": 50},
                        ax=axes[i,j])
            if(j != 0):
                axes[i,j].set_ylabel("")

            j = 1 if i == 1 else j
            i = 0 if i == 1 else i + 1

        fig.subplots_adjust(wspace=0.1)
        title = tw.fill(f"{name} - Comparison between amino acid and factor "
                        f"frequencies for genetic code: {code_name}", 100)
        fig.suptitle(title, fontsize=15, y=0.95)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        for ext in ["svg", "pdf"]:
            plt.savefig(f"{folder}/cor_regression_plot.{ext}",
                        bbox_inches="tight")

        plt.close()

        '''for aa in amino_acids:
            pair_df = pd.DataFrame({"Amino acid frequency": data_df[aa],
                                    "log10-Length": np.log10(data_df["Length"]),
                                    "GC": data_df["GC"],
                                    "Codon+GC": data_df[f"{aa}_gc"],
                                    "Codon+GC+Cost": data_df[f"{aa}_ener_gc"]})

            g = sns.pairplot(pair_df, corner=True, diag_kws={"color": "orange"})
            # Loop through each subplot and add grid to non-diagonal plots
            for i, j in zip(*np.tril_indices_from(g.axes, -1)):
                # Add grid to the non-diagonal plots
                g.axes[i, j].grid(True)

            title = tw.fill(f"{name} - Pairwise comparison between amino acid "
                            f"and factors for amino acid $\\mathbf{{{aa}}}$ "
                            f"and genetic code: {code_name}", 100)
            g.fig.suptitle(title, fontsize=15)
            g.fig.set_figheight(10)
            g.fig.set_figwidth(15)
            for ext in ["svg", "pdf"]:
                plt.savefig(f"{folder}/{aa}_cor_matrix_plot.{ext}",
                            bbox_inches="tight")

            plt.close()'''

        # Amino acid correlation data
        data_df = pd.read_csv(os.path.join(folder, "amino_acid_cor_data.csv"),
                              sep="\t", index_col=0)

        fig, axes = plt.subplots(1, 2)
        for i,cor in enumerate(["Pearson", "Spearman"]):
            pivot_df = data_df.pivot(index="Amino_Acid", columns="Factor",
                                     values=f"{cor}_Correlation")
            pivot_df = pivot_df[["Length", "GC", "Codon+GC", "Codon+GC+Cost"]]
            pivot_df = pivot_df.reindex(amino_acids)

            bar = True
            if(i==0):
                bar= False

            sns.heatmap(pivot_df, annot=True, cmap="coolwarm", center=0,
                        cbar=bar, vmin=-1, vmax=1, ax=axes[i])
            axes[i].set_ylabel("")
            axes[i].set_title(f"{cor} correlation coefficient")

        axes[0].set_ylabel("Amino acid")
        fig.subplots_adjust(wspace=0.05)
        title = tw.fill(f"{name} - Correlation of amino acid frequency and "
                        f"proteome factors for genetic code: {code_name}", 100)
        fig.suptitle(title, fontsize=15, y=0.96)
        fig.set_figheight(10)
        fig.set_figwidth(15)
        for ext in ["svg", "pdf"]:
            plt.savefig(f"{folder}/aa_cor_heatmap.{ext}", bbox_inches="tight")

        plt.close()
