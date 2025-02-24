import os
import sys
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import textwrap as tw
import scipy.stats as sci
import collections as defcol
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
import matplotlib.patches as patch


def optimal_bin(data):
    data = data.to_numpy()
    iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    h = 2 * iqr / len(data)**(1/3)
    return int((data.max() - data.min()) / h + 1)


def mean_corr(corrs, p_values):
    # Fisher's Z-transformation
    z_values = [0.5*np.log((1+r)/(1-r)) for r in corrs]
    mean_z = np.mean(z_values)
    mean_corr = (np.exp(2*mean_z)-1) / (np.exp(2*mean_z)+1)
    return [mean_corr, np.mean(p_values)]


def plot_lengths(data, kingdom, output):
    lengths = np.log10(data["Length_mean"])
    bins = optimal_bin(lengths)
    sns.histplot(lengths, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{kingdom} - Density of mean protein log10-length")
    plt.xlabel("Protein log10-length")
    plt.ylabel("Density")
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"protein_lengths.{ext}"),
                    bbox_inches="tight")

    plt.close()


def plot_gcs(data, kingdom, output):
    gcs = data["GC_mean"]
    bins = optimal_bin(gcs)
    sns.histplot(gcs, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{kingdom} - Density of mean protein GC content")
    plt.xlabel("GC content")
    plt.ylabel("Density")
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"protein_gcs.{ext}"),
                    bbox_inches="tight")

    plt.close()


def plot_amount(data, kingdom, output):
    nums = np.log10(data["#Proteins"])
    bins = optimal_bin(nums)
    sns.histplot(nums, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{kingdom} - Density of protein log10-number")
    plt.xlabel("log10-Number of proteins")
    plt.ylabel("Density")
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"protein_amount.{ext}"),
                    bbox_inches="tight")

    plt.close()


def plot_pct(data, aa_groups, kingdom, output):
    fig,axes = plt.subplots(2, 2)
    i = 0
    j = 0
    for aa_type,aa_list in aa_groups.items():
        aa_pct_code_cols = [f"{aa}_pct_code" for aa in aa_list]
        x_pos = np.arange(len(aa_list)) - 0.17
        c = "royalblue"
        code_box = axes[i,j].boxplot(data[aa_pct_code_cols], positions=x_pos,
                                     widths=0.3, notch=True, patch_artist=True,
                                     boxprops=dict(facecolor=c, color="black"),
                                    capprops=dict(color=c), whiskerprops=dict(color=c),
                                    flierprops=dict(color=c, markeredgecolor=c),
                                    medianprops=dict(color=c), zorder=2)

        aa_pct_freq_cols = [f"{aa}_pct_freq" for aa in aa_list]
        x_pos = np.arange(len(aa_list)) + 0.18
        c = "goldenrod"
        freq_box = axes[i,j].boxplot(data[aa_pct_freq_cols], positions=x_pos,
                                     widths=0.3,  notch=True, patch_artist=True,
                                     boxprops=dict(facecolor=c, color="black"),
                                     capprops=dict(color=c), whiskerprops=dict(color=c),
                                     flierprops=dict(color=c, markeredgecolor=c),
                                     medianprops=dict(color=c), zorder=2)

        axes[i,j].axhline(y=0, zorder=1, color="black")
        axes[i,j].grid(visible=True, which="major", color="#999999",
                       linestyle="dotted", alpha=0.5, zorder=0)
        axes[i,j].set_xticks(np.arange(len(aa_list)), aa_list)
        axes[i,j].set_xlabel("Amino acid")
        axes[i,j].set_title(f"{aa_type} amino acids")

        if(j == 0):
            axes[i,j].set_ylabel("Percentage change in %")
            if(i == 0):
                axes[i,j].legend([code_box["boxes"][0], freq_box["boxes"][0]],
                                 ["Codon number", "Codon+GC"],
                                 bbox_to_anchor=(1.45, 1.02), fancybox=True,
                                 fontsize=12)

        j = 1 if i == 1 else j
        i = 0 if i == 1 else i + 1

    spear_corr,spear_corr_p = mean_corr(data["Spearman_code"],
                                        data["Spearman_code_p"])
    kendall_corr,kendall_corr_p = mean_corr(data["Kendall_code"],
                                            data["Kendall_code_p"])
    axes[0,0].text(1.03, 0.6, f"Codon number\n"
        f"  - Spearman:\n"
        f"    - Coefficient: {spear_corr:.5f}\n"
        f"    - p-value: {spear_corr_p:.3e}\n"
        f"\n  - Kendall's Tau:\n"
        f"    - Coefficient: {kendall_corr:.5f}\n"
        f"    - p-value: {kendall_corr_p:.3e}",
        transform=axes[0,0].transAxes, fontsize=11,
        verticalalignment="top", linespacing=1.5,
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="grey", alpha=0.5))

    spear_corr,spear_corr_p = mean_corr(data["Spearman_freq"],
                                        data["Spearman_freq_p"])
    kendall_corr,kendall_corr_p = mean_corr(data["Kendall_freq"],
                                            data["Kendall_freq_p"])
    axes[0,0].text(1.03, 0.0, f"Codon+GC\n"
        f"  - Spearman:\n"
        f"    - Coefficient: {spear_corr:.5f}\n"
        f"    - p-value: {spear_corr_p:.3e}\n"
        f"\n  - Kendall's Tau:\n"
        f"    - Coefficient: {kendall_corr:.5f}\n"
        f"    - p-value: {kendall_corr_p:.3e}",
        transform=axes[0,0].transAxes, fontsize=11,
        verticalalignment="top", linespacing=1.5,
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="grey", alpha=0.5))

    fig.subplots_adjust(wspace=0.6, hspace=0.3)
    title = f"{kingdom} - Percentage change between amino acid frequencies"
    fig.suptitle(title, fontsize=15, y=0.95)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"pct_change.{ext}"),
                    bbox_inches="tight")

    plt.close()


def plot_corr_coefficients(data, output):
    color_palette = {"CN-spear": "maroon", "GC-spear": "darkorange",
                     "CN-kendall": "royalblue", "GC-kendall": "forestgreen"}
    data["Comb_col"] = data["Correlation"] + "-" + data["Comparison"]
    g = sns.FacetGrid(data, row="Kingdom", hue="Comb_col",
                      palette=list(color_palette.values()))
    g.map(sns.kdeplot, "Coefficient", clip_on=False, fill=False, alpha=1,
          color="black")
    g.map(sns.kdeplot, "Coefficient", clip_on=False, fill=True, alpha=0.5,
          hatch="x")
    g.refline(y=0, linewidth=2, linestyle="-", color="grey", clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, x.iloc[0], color="black", fontsize=13,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "Kingdom")
    g.set_titles("")
    g.set_xlabels(label="Correlation coefficient", fontsize=14)
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    legend_patches = [
        patch.Patch(color=color_palette["CN-spear"],
                    label="Spearman: Codon number"),
        patch.Patch(color=color_palette["GC-spear"],
                    label="Spearman: Codon+GC"),
        patch.Patch(color=color_palette["CN-kendall"],
                    label="Kendall's Tau: Codon number"),
        patch.Patch(color=color_palette["GC-kendall"],
                    label="Kendall's Tau: Codon+GC")
    ]
    plt.legend(handles=legend_patches, bbox_to_anchor=(0.53, 4.88), ncols=2)

    g.fig.set_figheight(10)
    g.fig.set_figwidth(15)

    title = "Correlation coefficient densities across kingdoms"
    plt.suptitle(title, x=0.55, y=1.08, fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(output, f"corr_coefficients.{ext}"),
                    bbox_inches="tight")

    plt.close()




# main method
if __name__ == "__main__":
    input = sys.argv[1]

    # Canonical amino acids order
    amino_acids = ["M", "W", "C", "D", "E", "F", "H", "K", "N", "Q", "Y", "I",
                   "A", "G", "P", "T", "V", "L", "R", "S"]
    aa_mean_cols = [f"{aa}_mean" for aa in amino_acids]
    aa_std_cols = [f"{aa}_std" for aa in amino_acids]
    corr_cols = ["Spearman_code", "Spearman_code_p", "Spearman_freq",
                 "Spearman_freq_p", "Kendall_code", "Kendall_code_p",
                 "Kendall_freq", "Kendall_freq_p"]
    aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
                 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
                 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}

    kingdoms = ["Archaea", "Bacteria", "Eukaryotes", "Viruses"]
    kingdom_freq_df = pd.DataFrame(columns=aa_mean_cols+aa_std_cols)
    kingdom_corr_df = pd.DataFrame(columns=["Coefficient", "Correlation",
                                            "Comparison", "Kingdom"])
    for kingdom in kingdoms:
        king_path = os.path.join(input, kingdom)
        data = pd.read_csv(os.path.join(king_path, "aa_corr_results.csv"),
                           sep="\t", header=0, index_col=0)
        plot_lengths(data, kingdom, king_path)
        plot_gcs(data, kingdom, king_path)
        plot_amount(data, kingdom, king_path)

        plot_pct(data, aa_groups, kingdom, king_path)

        # Calculate weighted mean of all amino acids frequencies
        kingdom_freq_df.loc[kingdom, aa_mean_cols] = data[aa_mean_cols].mul(data["#Proteins"], axis=0).sum() / data["#Proteins"].sum()
        # Calculate weighted standard deviation of all amino acids frequencies
        kingdom_freq_df.loc[kingdom, aa_std_cols] = np.sqrt(data[aa_std_cols].pow(2).mul(data["#Proteins"]-1, axis=0).sum() / (data["#Proteins"].sum()-data.shape[0]))

        for corr_type in ["Spearman", "Kendall"]:
            for comp_type in ["code", "freq"]:#
                local_corr_df = pd.DataFrame(columns=["Coefficient", "Correlation",
                                                      "Comparison", "Kingdom"])
                local_corr_df.loc[:, "Coefficient"] = data[f"{corr_type}_{comp_type}"]
                local_corr_df.loc[:, "Correlation"] = [corr_type] * len(data)
                local_corr_df.loc[:, "Comparison"] = [comp_type] * len(data)
                local_corr_df.loc[:, "Kingdom"] = [kingdom] * len(data)
                kingdom_corr_df = pd.concat([kingdom_corr_df if not kingdom_corr_df.empty else None,
                                             local_corr_df])

    ############################################################################
    kingdom_freq_df = kingdom_freq_df.T
    kingdom_freq_df.to_csv(os.path.join(input, "kingdom_abundances.csv"),
                           sep="\t")
    kingdom_freq_df.loc[aa_mean_cols].plot(kind="bar", yerr=kingdom_freq_df.loc[aa_std_cols].values.T,
                                           capsize=1, figsize=(14, 7), zorder=2)
    plt.xticks(np.arange(len(amino_acids)), amino_acids)
    plt.ylim(bottom=0)
    plt.xlabel("Amino acid")
    plt.ylabel("Amino acid frequency")
    plt.title("Mean protein amino acid frequencies across kingdoms")
    plt.legend(title="Kingdom", loc="upper left")
    plt.xticks(rotation=0)
    plt.grid(alpha=0.5, zorder=0)
    for ext in ["svg", "pdf"]:
        plt.savefig(os.path.join(input, f"amino_acid_freqs.{ext}"),
                    bbox_inches="tight")

    plt.close()
    ############################################################################

    plot_corr_coefficients(kingdom_corr_df, input)
