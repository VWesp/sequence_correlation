import os
import sys
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import textwrap as tw
import collections as defcol
from scipy.stats import norm
from Bio.Data import CodonTable
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

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


def plot_bar(prot_data, code_df, freq_df, aa_groups, kingdom, code_abbr):
    freq_mean = freq_df.mean(axis=0)
    freq_std = freq_df.std(axis=0)

    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 2)
    i = 0
    j = 0
    for aa_type,aa_list in aa_groups.items():
        x_pos = np.arange(len(aa_list)) - 0.25
        axes[i,j].bar(x_pos, prot_data[1][aa_list], yerr=prot_data[2][aa_list],
                      width=0.25, color=cmap(0.3), edgecolor="black", linewidth=0.75,
                      label="Observed", capsize=3, zorder=2)

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
            axes[i,j].set_ylabel("Mean amino acid frequencies")

        j = 1 if i == 1 else j
        i = 0 if i == 1 else i + 1

    y_max = max(max([ax.get_ylim() for ax in axes.reshape(-1)]))
    for ax in axes.reshape(-1):
        ax.set_ylim(0, y_max)

    axes[0,0].legend(bbox_to_anchor=(1.3, -0.05), fancybox=True, fontsize=12)
    fig.subplots_adjust(hspace=0.4)
    title = tw.fill(f"{kingdom} - Mean proteome amino acid frequency "
                    f"for genetic code: {code_abbr}", 100)
    fig.suptitle(title, fontsize=15, y=0.95)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    plt.show()
    plt.close()


def plot_protein_lengths(prot_data, kingdom):
    lengths = prot_data["Length"]
    bins = optimal_bin(lengths)
    sns.histplot(lengths, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{kingdom} - Density of mean proteome length")
    plt.xlabel("Protein length")
    plt.ylabel("Count")
    plt.show()
    plt.close()


def plot_protein_gcs(prot_data, kingdom):
    gcs = prot_data["GC"]
    bins = optimal_bin(gcs)
    sns.histplot(gcs, bins=bins, alpha=0.4, color="maroon", kde=True,
                 line_kws={"linewidth": 2, "linestyle": "--"})
    plt.title(f"{kingdom} - Density of mean proteome length")
    plt.xlabel("Protein length")
    plt.ylabel("Count")
    plt.show()
    plt.close()


def plot_ridge(corr_data, kingdom):
    


if __name__ == "__main__":
    path = sys.argv[1]

    aa_groups = {"Aliphatic": ["A", "G", "I", "L", "M", "V"], "Aromatic": ["F",
                 "W", "Y"], "Charged": ["D", "E", "H", "K", "R"],
                 "Uncharged": ["C", "N", "P", "Q", "S", "T"]}

    kingdoms = ["Archaea", "Bacteria", "Eukaryota", "Viruses"]
    kingdoms_data = {}
    for kingdom in kingdoms:
        king_path = os.path.join(path, kingdom.lower())
        prot_df = pd.read_csv(os.path.join(king_path, "proteome_data.csv"), sep="\t",
                              header=0, index_col=0)

        #plot_protein_lengths(prot_df, kingdom)
        #plot_protein_gcs(prot_df, kingdom)

        num_prots = prot_df["#Proteins"]
        weighted_means = prot_df.mul(num_prots, axis=0).sum() / num_prots.sum()
        weighted_std = np.sqrt(((prot_df-weighted_means)**2).mul(num_prots, axis=0).sum() / num_prots.sum())
        kingdoms_data[kingdom] = [prot_df, weighted_means, weighted_std]

        gen_cod_folders = [os.path.join(king_path, folder) for folder in os.listdir(king_path)
                           if os.path.isdir(os.path.join(king_path, folder))]
        code_corr_data = {}
        for cod_fold in gen_cod_folders:
            code_basename = os.path.basename(cod_fold)
            code_abbr = CODE_ABBREVIATIONS_INV[code_basename]

            freq_df = pd.read_csv(os.path.join(cod_fold, "pred_freq_data.csv"),
                                  sep="\t", header=0, index_col=0)

            code_df = pd.read_csv(os.path.join(cod_fold, "norm_code_data.csv"),
                                 sep="\t", header=0, index_col=0)

            #plot_bar(kingdoms_data[kingdom], code_df, freq_df,
            #             aa_groups, kingdom, code_abbr)

            corr_df = pd.read_csv(os.path.join(cod_fold, "corr_data.csv"),
                                  sep="\t", header=0, index_col=0)
            code_corr_data[code_abbr] = corr_df

        plot_ridge(code_corr_data, kingdom)
        dsjaiodhas
