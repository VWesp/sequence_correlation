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

# abbreviations for each genetic code
CODE_ABBREVIATIONS = {"AFM": "alternative_flatworm_mitochondrial", "AY": "alternative_yeast",
                      "AM": "ascidian_mitochondrial", "BAPP": "bacterial-archaeal-plant_plastid",
                      "BN": "blastocrithidia_nuclear", "CDSG": "candidate_division_sr1_and_gracilibacteria",
                      "CMUT": "cephalodiscidae_mitochondrial_uaa-tyr", "CM": "chlorophycean_mitochondrial",
                      "CDHN": "ciliate-dasycladacean-hexamita_nuclear", "CN": "condylostoma_nuclear",
                      "EFM": "echinoderm-flatworm_mitochondrial", "EN": "euplotid_nuclear",
                      "IM": "invertebrate_mitochondrial", "KN": "karyorelict_nuclear",
                      "MN": "mesodinium_nuclear", "MPCMMS": "mold-protozoan-coelenterate_mitochondrial_and_mycoplasma_spiroplasma",
                      "PTN": "pachsyolen_tannophilus_nuclear", "PN": "peritrich_nuclear",
                      "RM": "rhabdopleuridae_mitochondrial", "SOM": "scenedesmus_obliquus_mitochondrial",
                      "SGCode": "standard", "TM": "thraustochytrium_mitochondrial",
                      "TrM": "trematode_mitochondrial", "VM": "vertebrate_mitochondrial",
                      "YM": "yeast_mitochondrial"}
CODE_ABBREVIATIONS_INV = {v:k for k,v in CODE_ABBREVIATIONS.items()}


def optimal_bin(data):
    data = data.to_numpy()
    iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    h = 2 * iqr / len(data)**(1/3)
    return int((data.max() - data.min()) / h + 1)


def plot_lengths(prot_data, kingdom, output):
    lengths = prot_data["Length"]
    bins = optimal_bin(lengths)
    sns.histplot(lengths, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{kingdom} - Density of mean protein length")
    plt.xlabel("Protein length")
    plt.ylabel("Density")
    for ext in ["svg", "pdf"]:
        plt.savefig(f"{output}/proteomic_lengths.{ext}", bbox_inches="tight")

    plt.close()


def plot_gcs(prot_data, kingdom, output):
    gcs = prot_data["GC"]
    bins = optimal_bin(gcs)
    sns.histplot(gcs, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{kingdom} - Density of mean protein GC content")
    plt.xlabel("GC content")
    plt.ylabel("Density")
    for ext in ["svg", "pdf"]:
        plt.savefig(f"{output}/proteomic_gcs.{ext}", bbox_inches="tight")

    plt.close()


def plot_amount(prot_data, kingdom, output):
    nums = prot_data["#Proteins"]
    bins = optimal_bin(nums)
    sns.histplot(nums, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{kingdom} - Density of protein number")
    plt.xlabel("Number of proteins")
    plt.ylabel("Density")
    for ext in ["svg", "pdf"]:
        plt.savefig(f"{output}/proteomic_amount.{ext}", bbox_inches="tight")

    plt.close()


def plot_bar(prot_data, code_df, freq_df, corr_code, corr_gc, aa_groups,
             kingdom, code_abbr, output):
    prot_mean = prot_data["mean"][kingdom]
    prot_std = prot_data["std"][kingdom]

    freq_mean = freq_df.mean(axis=0)
    freq_std = freq_df.std(axis=0)

    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 2)
    i = 0
    j = 0
    for aa_type,aa_list in aa_groups.items():
        x_pos = np.arange(len(aa_list)) - 0.25
        axes[i,j].bar(x_pos, prot_mean[aa_list], yerr=prot_std[aa_list],
                      width=0.25, color=cmap(0.3), edgecolor="black",
                      linewidth=0.75, label="Observed", capsize=3, zorder=2)

        x_pos = np.arange(len(aa_list))
        axes[i,j].bar(x_pos, code_df["frequency"][aa_list], width=0.25,
                      color=cmap(0.6), edgecolor="black", linewidth=0.75,
                      label="Codon number", capsize=3, zorder=2)

        x_pos = np.arange(len(aa_list)) + 0.25
        axes[i,j].bar(x_pos, freq_mean[aa_list], yerr=freq_std[aa_list],
                      width=0.25, color=cmap(0.9), edgecolor="black",
                      linewidth=0.75, label="Codon+GC", capsize=3, zorder=2)

        axes[i,j].grid(visible=True, which="major", color="#999999",
                       linestyle="dotted", alpha=0.5, zorder=0)
        axes[i,j].set_xticks(np.arange(len(aa_list)), aa_list)
        axes[i,j].set_xlabel("Amino acid")
        axes[i,j].set_title(f"{aa_type} amino acids")

        if(j == 0):
            axes[i,j].set_ylabel("Mean amino acid frequency")

        j = 1 if i == 1 else j
        i = 0 if i == 1 else i + 1

    y_max = max(max([ax.get_ylim() for ax in axes.reshape(-1)]))
    for ax in axes.reshape(-1):
        ax.set_ylim(0, y_max)

    axes[0,0].legend(bbox_to_anchor=(1.45, 1.02), fancybox=True, fontsize=12)

    axes[0,0].text(1.03, 0.6, f"Codon number\n"
        f"  - Spearman:\n"
        f"    - Coefficient: {corr_code[0][0]:.5f}\n"
        f"    - p-value: {corr_code[0][1]:.3e}\n"
        f"\n  - Kendall's Tau:\n"
        f"    - Coefficient: {corr_code[1][0]:.5f}\n"
        f"    - p-value: {corr_code[1][1]:.3e}",
        transform=axes[0,0].transAxes, fontsize=11,
        verticalalignment="top", linespacing=1.5,
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="grey", alpha=0.5))

    axes[0,0].text(1.03, 0.0, f"Codon+GC\n"
        f"  - Spearman:\n"
        f"    - Coefficient: {corr_gc[0][0]:.5f}\n"
        f"    - p-value: {corr_gc[0][1]:.3e}\n"
        f"\n  - Kendall's Tau:\n"
        f"    - Coefficient: {corr_gc[1][0]:.5f}\n"
        f"    - p-value: {corr_gc[1][1]:.3e}",
        transform=axes[0,0].transAxes, fontsize=11,
        verticalalignment="top", linespacing=1.5,
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="grey", alpha=0.5))

    fig.subplots_adjust(wspace=0.6, hspace=0.3)
    title = tw.fill(f"{kingdom} - Mean proteomic amino acid frequencies "
                    f"for genetic code: {code_abbr}", 100)
    fig.suptitle(title, fontsize=15, y=0.95)

    fig.set_figheight(10)
    fig.set_figwidth(15)
    for ext in ["svg", "pdf"]:
        plt.savefig(f"{code_folder}/corr_bar_plot.{ext}", bbox_inches="tight")

    plt.close()


def mean_corr(corrs, p_values):
    z_values = [0.5*np.log((1+r)/(1-r)) for r in corrs]
    mean_z = np.mean(z_values)
    mean_corr = (np.exp(2*mean_z)-1) / (np.exp(2*mean_z)+1)

    chi_squared = -2 * np.sum([np.log(p+np.nextafter(0, 1)) for p in p_values])
    dof = 2 * len(p_values)
    mean_p = sci.chi2.sf(chi_squared, df=dof)

    return [mean_corr, max(np.nextafter(0, 1), mean_p)]


def plot_ridge(corr_df, corr_type, kingdom, output):
    corr_df = corr_df[corr_df["c_type"]==corr_type].copy()
    min_val = np.min(corr_df["corr"]) * 0.99
    max_val = np.max(corr_df["corr"]) * 1.01

    color_palette = {"CN": "royalblue", "GC": "darkorange"}
    g = sns.FacetGrid(corr_df, row="code", hue="a_type",
                      palette=list(color_palette.values()))
    g.map(sns.kdeplot, "corr", clip_on=False, fill=False, alpha=1, color="black")
    g.map(sns.kdeplot, "corr", clip_on=False, fill=True, alpha=0.7)
    g.refline(y=0, linewidth=2, linestyle="-", color="grey", clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, x.iloc[0], color="black", fontsize=13,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "code")
    g.set_titles("")
    g.set_xlabels(label="Correlation coefficient", fontsize=14)
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    legend_patches = [
        patch.Patch(color=color_palette["CN"], label="Codon number"),
        patch.Patch(color=color_palette["GC"], label="Codon+GC")
    ]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1, 30))

    g.fig.set_figheight(10)
    g.fig.set_figwidth(15)

    corr_type = corr_df["c_type"].iloc[0]
    corr_name = corr_df["c_name"].iloc[0]
    title = tw.fill(f"{kingdom} - {corr_name} correlation coefficient densities "
                    f"for all genetic codes", 65)
    plt.suptitle(title, x=0.55, y=1.05, fontsize=18)
    for ext in ["svg", "pdf"]:
        plt.savefig(f"{output}/{corr_type}_corr_ridgeplot.{ext}",
                    bbox_inches="tight")

    plt.close()


def plot_scatterplot(corr_df, corr_type, comp_df, comp_type, kingdom, output):
    corr_df = corr_df[corr_df["c_type"]==corr_type].copy()

    codon_corrs = corr_df[corr_df["a_type"]=="codon"].copy()
    codon_corrs.loc[:,comp_type] = comp_df[comp_type]
    g = sns.JointGrid(data=codon_corrs, x=comp_type, y="corr")
    ax_codon = sns.scatterplot(data=codon_corrs, x=comp_type, y="corr",
                               alpha=0.5, color="royalblue", ax=g.ax_joint,
                               linewidth=1, edgecolor="black",
                               label="Codon number")

    gc_corrs = corr_df[corr_df["a_type"]=="gc"].copy()
    gc_corrs.loc[:,comp_type] = comp_df[comp_type]
    ax_gc = sns.scatterplot(data=gc_corrs, x=comp_type, y="corr", alpha=0.5,
                            color="darkorange", ax=g.ax_joint, linewidth=1,
                            edgecolor="black", label="Codon+GC")

    sns.regplot(data=codon_corrs, x=comp_type, y="corr", scatter=False,
                ax=ax_codon)
    sns.regplot(data=gc_corrs, x=comp_type, y="corr", scatter=False, ax=ax_gc)

    sns.histplot(data=gc_corrs[comp_type], ax=g.ax_marg_x, color="purple",
                 element="step", linewidth=1, edgecolor="black")
    hist_y1, bins_y1 = np.histogram(codon_corrs["corr"], density=True,
                                    bins=optimal_bin(codon_corrs["corr"]))
    g.ax_marg_y.fill_betweenx(bins_y1[:-1], 0, hist_y1, step="pre",
                              color="royalblue", alpha=0.5, linewidth=1,
                              edgecolor="black")
    hist_y2, bins_y2 = np.histogram(gc_corrs["corr"], density=True,
                                    bins=optimal_bin(gc_corrs["corr"]))
    g.ax_marg_y.fill_betweenx(bins_y2[:-1], 0, hist_y2, step="pre",
                              color="darkorange", alpha=0.5, linewidth=1,
                              edgecolor="black")

    x_label = "Number of proteins"
    if(comp_type == "GC"):
        x_label = "Proteomic GC content"
    elif(comp_type == "Length"):
        x_label = "Protein length"

    g.ax_joint.set_xlabel(x_label)
    g.ax_joint.set_ylabel(f"Correlation coefficient")
    corr_type = corr_df["c_type"].iloc[0]
    corr_name = corr_df["c_name"].iloc[0]
    code = corr_df["code"].iloc[0]
    g.fig.suptitle(f"{kingdom} - {corr_name} correlation coefficients "
                   f"for genetic code: {code}")
    sns.move_legend(g.ax_joint, "lower right")
    g.fig.subplots_adjust(top=0.92)
    g.fig.set_figheight(10)
    g.fig.set_figwidth(15)

    for ext in ["svg", "pdf"]:
        plt.savefig(f"{code_folder}/{corr_type}_vs_{comp_type.lower()}_scatterplot.{ext}",
                    bbox_inches="tight")

    plt.close()


def pct_change(s1, s2):
    return ((s1 - s2) / s1) * 100


def get_pct_change_data(s1, s2, s3):
    code_pct = pct_change(s1, s2)
    code_pct.index = [f"{idx}_code" for idx in code_pct.index]
    freq_pct = pct_change(s1, s3)
    freq_pct.index = [f"{idx}_gc" for idx in freq_pct.index]
    both_pct_df = pd.concat([code_pct, freq_pct]).reindex([i for pair in zip(code_pct.index, freq_pct.index)
                                                           for i in pair])
    return both_pct_df


if __name__ == "__main__":
    path = sys.argv[1]

    aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
                 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
                 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}

    kingdoms = ["Archaea", "Bacteria", "Eukaryota", "Viruses"]
    kingdoms_freqs_data = {"mean": {}, "std": {}}
    for kingdom in kingdoms:
        print(f"Printing stuff for {kingdom}...")
        king_path = os.path.join(path, kingdom.lower())
        prot_df = pd.read_csv(os.path.join(king_path, "proteome_mean_data.csv"),
                              sep="\t", header=0, index_col=0)
        prot_std_df = pd.read_csv(os.path.join(king_path, "proteome_std_data.csv"),
                                  sep="\t", header=0, index_col=0)

        prot_mean_data = pd.Series()
        prot_std_data = pd.Series()
        for col in prot_df.columns:
            prot_mean_data[col] = (prot_df[col] * prot_df["#Proteins"]).sum() / prot_df["#Proteins"].sum()
            prot_std_data[col] = np.sqrt((prot_std_df[col]**2 * (prot_std_df["#Proteins"]-1)).sum() / (prot_std_df["#Proteins"]-1).sum())

        kingdoms_freqs_data["mean"][kingdom] = prot_mean_data
        kingdoms_freqs_data["std"][kingdom] = prot_std_data

        plot_lengths(prot_df, kingdom, king_path)
        plot_gcs(prot_df, kingdom, king_path)
        plot_amount(prot_df, kingdom, king_path)

        gen_code_folders = [os.path.join(king_path, folder)
                            for folder in os.listdir(king_path)
                            if os.path.isdir(os.path.join(king_path, folder))]
        kingdom_corrs_data=pd.DataFrame(columns=["code_spearman", "code_p_spearman",
                                                 "code_kendall", "code_p_kendall",
                                                 "gc_spearman", "gc_p_spearman",
                                                 "gc_kendall", "gc_p_kendall"])
        all_corrs_df = pd.DataFrame()
        all_pct_df = pd.DataFrame()
        for code_folder in gen_code_folders:
            code_basename = os.path.basename(code_folder)
            code_abbr = CODE_ABBREVIATIONS_INV[code_basename]
            print(f"\tPrinting stuff for {code_abbr}...")

            code_df = pd.read_csv(os.path.join(code_folder, "norm_code_data.csv"),
                                 sep="\t", header=0, index_col=0)

            freq_df = pd.read_csv(os.path.join(code_folder, "pred_freq_data.csv"),
                                  sep="\t", header=0, index_col=0)

            corr_code_df = pd.read_csv(os.path.join(code_folder, "corr_code_data.csv"),
                                       sep="\t", header=0, index_col=0)
            code_spearman = mean_corr(corr_code_df["spearman"],
                                      corr_code_df["p_spearman"])
            kingdom_corrs_data.loc[code_abbr, "code_spearman"] = code_spearman[0]
            kingdom_corrs_data.loc[code_abbr, "code_p_spearman"] = code_spearman[1]

            code_kendall = mean_corr(corr_code_df["kendall"],
                                     corr_code_df["p_kendall"])
            kingdom_corrs_data.loc[code_abbr, "code_kendall"] = code_kendall[0]
            kingdom_corrs_data.loc[code_abbr, "code_p_kendall"] = code_kendall[1]

            corr_gc_df = pd.read_csv(os.path.join(code_folder, "corr_gc_data.csv"),
                                     sep="\t", header=0, index_col=0)
            gc_spearman = mean_corr(corr_gc_df["spearman"],
                                    corr_gc_df["p_spearman"])
            kingdom_corrs_data.loc[code_abbr, "gc_spearman"] = gc_spearman[0]
            kingdom_corrs_data.loc[code_abbr, "gc_p_spearman"] = gc_spearman[1]

            gc_kendall = mean_corr(corr_gc_df["kendall"],
                                   corr_gc_df["p_kendall"])
            kingdom_corrs_data.loc[code_abbr, "gc_kendall"] = gc_kendall[0]
            kingdom_corrs_data.loc[code_abbr, "gc_p_kendall"] = gc_kendall[1]

            plot_bar(kingdoms_freqs_data, code_df, freq_df, [code_spearman,
                     code_kendall], [gc_spearman, gc_kendall], aa_groups,
                     kingdom, code_abbr, code_folder)

            all_pct_df[code_abbr] = get_pct_change_data(kingdoms_freqs_data["mean"][kingdom].iloc[3:],
                                                        code_df["frequency"], freq_df.mean())

            corrs_df = pd.DataFrame()
            corr_types = {"codon": corr_code_df, "gc": corr_gc_df}
            for corr_type in ["spearman", "kendall"]:
                for a_type,df in corr_types.items():
                    c_name = "Spearman" if corr_type=="spearman" else "Kendall's Tau"
                    local_corrs = pd.DataFrame({"corr": df[corr_type],
                                  "a_type": a_type, "c_type": corr_type,
                                  "c_name": c_name, "code": code_abbr})
                    corrs_df = pd.concat([corrs_df, local_corrs])

            all_corrs_df = pd.concat([all_corrs_df, corrs_df])

            for corr_type in ["spearman", "kendall"]:
                for comp_type in ["GC", "Length", "#Proteins"]:
                    plot_scatterplot(corrs_df, corr_type, prot_df, comp_type,
                                     kingdom, code_folder)

        plot_ridge(all_corrs_df, "spearman", kingdom, king_path)
        plot_ridge(all_corrs_df, "kendall", kingdom, king_path)

        kingdom_corrs_data.to_csv(os.path.join(king_path, "code_correlation_data.csv"),
                                  sep="\t")
        all_pct_df.to_csv(os.path.join(king_path, "pct_change_data.csv"),
                          sep="\t")

    mean_data = pd.DataFrame(kingdoms_freqs_data["mean"])
    std_data = pd.DataFrame(kingdoms_freqs_data["std"])
    mean_data.iloc[3:].plot(kind="bar", yerr=std_data.iloc[3:], capsize=1,
                            figsize=(14, 7), zorder=2)
    plt.ylim(bottom=0)
    plt.xlabel("Amino acid")
    plt.ylabel("Mean distribution")
    plt.title("Mean proteomic amino acid distribution across kingdoms")
    plt.legend(title="Kingdom", loc="upper left")
    plt.xticks(rotation=0)
    plt.grid(alpha=0.5, zorder=0)
    for ext in ["svg", "pdf"]:
        plt.savefig(f"{path}/amino_acid_distribution.{ext}",
                    bbox_inches="tight")

    plt.close()

    std_data.index = [f"{idx}_std" for idx in std_data.index]
    combined_df = pd.concat([mean_data, std_data]).reindex([i for pair in zip(mean_data.index, std_data.index)
                                                            for i in pair])
    combined_df.to_csv(f"{path}/kingdom_data.csv", sep="\t")
